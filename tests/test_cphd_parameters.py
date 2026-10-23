"""Synthetic CPHD and integration checks; execute on the Linux server only."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

from tools.cphd_parameters import extract_parameters, make_product_prompt
from tools.target_statistics import generate_dataset


XML = """<CPHD xmlns="http://api.nsgreg.nga.mil/schema/cphd/1.0.1">
<CollectionID><CollectorName>TEST-SAT</CollectorName><CoreName>synthetic</CoreName>
<CollectType>MONOSTATIC</CollectType><RadarMode><ModeType>SPOTLIGHT</ModeType></RadarMode></CollectionID>
<Global><DomainType>FX</DomainType><Timeline><CollectionStart>2026-01-01T00:00:00Z</CollectionStart>
<TxTime1>0.25</TxTime1><TxTime2>5.25</TxTime2></Timeline></Global>
<Data><Channel><Identifier>CH1</Identifier><NumVectors>2000</NumVectors>
<NumSamples>4096</NumSamples></Channel></Data>
<Channel><RefChId>CH1</RefChId><Parameters><Identifier>CH1</Identifier>
<Polarization><TxPol>H</TxPol><RcvPol>V</RcvPol></Polarization>
<FxC>9650000000</FxC><FxBW>500000000</FxBW>
<TxRcv><TxWFId>TX1</TxWFId><RcvId>RX1</RcvId></TxRcv></Parameters></Channel>
<ReferenceGeometry><Monostatic><SideOfTrack>R</SideOfTrack><SlantRange>700000</SlantRange>
<IncidenceAngle>35.125</IncidenceAngle><GrazeAngle>54.875</GrazeAngle>
<AzimuthAngle>123.25</AzimuthAngle></Monostatic></ReferenceGeometry>
<TxRcv><TxWFParameters><Identifier>TX1</Identifier><PulseLength>0.00002</PulseLength>
<RFBandwidth>450000000</RFBandwidth><FreqCenter>9650000000</FreqCenter></TxWFParameters>
<RcvParameters><Identifier>RX1</Identifier><SampleRate>600000000</SampleRate>
<WindowLength>0.00003</WindowLength><IFFilterBW>550000000</IFFilterBW></RcvParameters></TxRcv>
</CPHD>"""


def write_cphd(path, xml=XML, version="1.0.1"):
    xml_bytes = xml.encode("utf-8")
    header = (f"CPHD/{version}\nXML_BLOCK_SIZE := {len(xml_bytes)}\n"
              "XML_BLOCK_BYTE_OFFSET := 4096\n\f\n").encode("ascii")
    # Padding and non-XML payload exercise the XML block's actual byte bounds.
    path.write_bytes(header.ljust(4096, b"\x00") + xml_bytes + b"\f\n\xffbinary signal data")


class CphdTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cphd = self.root / "product.cphd"
        write_cphd(self.cphd)

    def test_metadata_units_and_selected_channel_links(self):
        parameters = extract_parameters(self.cphd)
        self.assertEqual(parameters["channel"]["saved_center_frequency_hz"], 9650000000)
        self.assertEqual(parameters["channel"]["saved_bandwidth_hz"], 500000000)
        self.assertEqual(parameters["transmit_waveforms"][0]["rf_bandwidth_hz"], 450000000)
        self.assertEqual(parameters["receivers"][0]["sample_rate_hz"], 600000000)
        self.assertEqual(parameters["reference_geometry"]["incidence_angle_deg"], 35.125)
        self.assertEqual(parameters["collection"]["collection_start_utc"], "2026-01-01T00:00:00Z")
        prompt = make_product_prompt(parameters)
        for text in ("9.650000 GHz", "saved bandwidth: 500.000 MHz", "H/V",
                     "transmitted RF bandwidth: 450.000 MHz", "20.000 us",
                     "sample rate: 600.000 MHz", "product reference geometry (channel CH1)",
                     "right-looking", "35.125 deg", "700.000 km"):
            self.assertIn(text, prompt)
        for text in ("PRF", "resolution", "synthetic", str(self.cphd)):
            self.assertNotIn(text, prompt)
        self.assertNotIn("prompt", parameters)

    def test_optional_metadata_is_not_fabricated(self):
        xml = XML[:XML.index("<ReferenceGeometry>")] + "</CPHD>"
        xml = xml.replace("<TxRcv><TxWFId>TX1</TxWFId><RcvId>RX1</RcvId></TxRcv>", "")
        write_cphd(self.cphd, xml)
        parameters = extract_parameters(self.cphd)
        self.assertEqual(parameters["transmit_waveforms"], [])
        prompt = make_product_prompt(parameters)
        for absent in ("sample rate", "pulse length", "reference geometry", "None"):
            self.assertNotIn(absent, prompt)

    def test_multi_channel_requires_selection_and_uses_selected_parameters(self):
        extra_channel = """<Parameters><Identifier>CH2</Identifier>
<Polarization><TxPol>V</TxPol><RcvPol>V</RcvPol></Polarization>
<FxC>10000000000</FxC><FxBW>300000000</FxBW></Parameters>"""
        xml = XML.replace("</Parameters></Channel>", "</Parameters>" + extra_channel + "</Channel>")
        extra_data = "<Channel><Identifier>CH2</Identifier><NumVectors>500</NumVectors><NumSamples>600</NumSamples></Channel>"
        write_cphd(self.cphd, xml.replace("</Data>", extra_data + "</Data>"))
        with self.assertRaisesRegex(ValueError, "Multiple CPHD channels"):
            extract_parameters(self.cphd)
        parameters = extract_parameters(self.cphd, "CH2")
        self.assertEqual(parameters["channel"]["num_vectors"], 500)
        self.assertEqual(parameters["transmit_waveforms"], [])
        prompt = make_product_prompt(parameters)
        self.assertIn("10.000000 GHz", prompt)
        self.assertIn("V/V", prompt)
        self.assertIn("reference geometry (channel CH1)", prompt)
        with self.assertRaisesRegex(ValueError, "Unknown CPHD channel"):
            extract_parameters(self.cphd, "missing")

    def test_invalid_metadata_stops_instead_of_guessing(self):
        cases = (
            (XML, "0.3", "Unsupported CPHD"),
            (XML, "1.1.0", "namespace"),
            (XML.replace("9650000000", "NaN"), "1.0.1", "Non-finite"),
            (XML.replace("<FxBW>500000000</FxBW>", ""), "1.0.1", "Missing CPHD"),
            (XML.replace("<RcvId>RX1</RcvId>", "<RcvId>unknown</RcvId>"), "1.0.1", "Unresolved"),
        )
        for xml, version, message in cases:
            with self.subTest(message=message):
                write_cphd(self.cphd, xml, version)
                with self.assertRaisesRegex(ValueError, message):
                    extract_parameters(self.cphd)
        self.cphd.write_bytes(b"CPHD/1.0.1\nXML_BLOCK_SIZE := 9999\nXML_BLOCK_BYTE_OFFSET := 4096\n\f\n")
        with self.assertRaisesRegex(ValueError, "outside the file"):
            extract_parameters(self.cphd)

    def test_cphd_110_namespace(self):
        write_cphd(self.cphd, XML.replace("cphd/1.0.1", "cphd/1.1.0"), "1.1.0")
        self.assertEqual(extract_parameters(self.cphd)["cphd_version"], "1.1.0")

    def prepare_pairs(self):
        for directory in ("echo_5_new", "image_5_new"):
            (self.root / directory).mkdir()
        for name in ("a.mat", "b.mat"):
            (self.root / "echo_5_new" / name).touch()
            raw = np.ones((12, 12), dtype=[("real", "<f4"), ("imag", "<f4")])
            with h5py.File(self.root / "image_5_new" / name, "w") as handle:
                handle.create_dataset("pfa_patch", data=raw)

    def test_cli_appends_context_to_every_pair_and_writes_three_schemas(self):
        self.prepare_pairs()
        output = self.root / "outputs/dataset_pair.json"
        script = Path(__file__).resolve().parents[1] / "tools/target_statistics.py"
        subprocess.run([sys.executable, str(script), "--data-root", str(self.root),
                        "--output-json", str(output), "--cphd-file", str(self.cphd)],
                       check=True, capture_output=True, text=True)
        pairs = json.loads(output.read_text(encoding="utf-8"))
        metadata = json.loads((output.parent / "metadata.json").read_text(encoding="utf-8"))
        parameters = json.loads((output.parent / "cphd_parameters.json").read_text(encoding="utf-8"))
        context = make_product_prompt(parameters)
        self.assertEqual(len(pairs), 2)
        for pair, details in zip(pairs, metadata):
            self.assertEqual(set(pair), {"source", "target", "prompt"})
            self.assertEqual(set(details), {"source", "target", "image_attributes"})
            self.assertEqual(pair["source"], details["source"])
            self.assertTrue(pair["prompt"].startswith("Echo to image."))
            self.assertIn("Amplitude contrast:", pair["prompt"])
            self.assertTrue(pair["prompt"].endswith(context))
            self.assertEqual(pair["prompt"].count("CPHD product acquisition parameters:"), 1)

    def test_invalid_cphd_preserves_existing_outputs_and_input(self):
        self.prepare_pairs()
        output = self.root / "dataset_pair.json"
        paths = [output, self.root / "metadata.json", self.root / "cphd_parameters.json"]
        for path in paths:
            path.write_text("previous", encoding="utf-8")
        self.cphd.write_bytes(b"not CPHD")
        with self.assertRaises(ValueError):
            generate_dataset(data_root=self.root, output_json=output, cphd_file=self.cphd)
        self.assertTrue(all(path.read_text(encoding="utf-8") == "previous" for path in paths))
        for destination in (output, self.cphd):
            with self.assertRaises(ValueError):
                generate_dataset(data_root=self.root, output_json=output, cphd_file=self.cphd,
                                 cphd_parameters_json=destination)
        self.assertEqual(self.cphd.read_bytes(), b"not CPHD")

    def test_custom_pilot_parameters_and_failed_target_preserve_full_outputs(self):
        self.prepare_pairs()
        output = self.root / "dataset_pair.json"
        full_parameters = self.root / "cphd_parameters.json"
        full_parameters.write_text("full run", encoding="utf-8")
        pilot_parameters = self.root / "cphd_parameters_pilot.json"
        generate_dataset(data_root=self.root, output_json=output, cphd_file=self.cphd,
                         cphd_parameters_json=pilot_parameters, limit=1)
        self.assertEqual(full_parameters.read_text(encoding="utf-8"), "full run")
        self.assertEqual(json.loads(pilot_parameters.read_text(encoding="utf-8"))
                         ["channel"]["identifier"], "CH1")
        paths = [output, self.root / "metadata.json", pilot_parameters]
        previous = [path.read_bytes() for path in paths]
        (self.root / "image_5_new/b.mat").write_bytes(b"invalid target")
        with self.assertRaisesRegex(ValueError, "Record 1"):
            generate_dataset(data_root=self.root, output_json=output, cphd_file=self.cphd,
                             cphd_parameters_json=pilot_parameters)
        self.assertEqual([path.read_bytes() for path in paths], previous)


if __name__ == "__main__":
    unittest.main()
