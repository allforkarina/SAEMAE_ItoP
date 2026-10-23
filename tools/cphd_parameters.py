"""Read CPHD 1.0/1.1 XML parameters without loading signal or PVP arrays."""

import math
from pathlib import Path
from xml.etree import ElementTree as ET


def read_cphd_xml(path):
    """Locate the XML block using the CPHD ASCII header's byte offsets."""
    path = Path(path)
    with path.open("rb") as handle:
        signature = handle.readline().decode("ascii").strip()
        if signature not in ("CPHD/1.0.0", "CPHD/1.0.1", "CPHD/1.1.0"):
            raise ValueError(f"Unsupported CPHD version: {signature!r}; expected CPHD 1.0/1.1")
        header = {}
        while True:
            line = handle.readline()
            if line == b"\f\n":
                break
            if not line or b" := " not in line:
                raise ValueError("Invalid CPHD ASCII header or missing section terminator")
            name, value = line.decode("ascii").strip().split(" := ", 1)
            header[name] = value
        offset = int(header["XML_BLOCK_BYTE_OFFSET"])
        size = int(header["XML_BLOCK_SIZE"])
        if offset < handle.tell() or size <= 0 or offset + size > path.stat().st_size:
            raise ValueError("CPHD XML block lies outside the file or overlaps the header")
        handle.seek(offset)
        xml = handle.read(size)
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as error:
        raise ValueError(f"Invalid CPHD XML: {error}") from error
    version = signature.split("/", 1)[1]
    namespace = f"http://api.nsgreg.nga.mil/schema/cphd/{version}"
    if root.tag != f"{{{namespace}}}CPHD":
        raise ValueError("CPHD XML root/namespace does not match the file version")
    # All supported fields belong to the CPHD namespace; simplify direct paths.
    for node in root.iter():
        if node.tag.startswith(f"{{{namespace}}}"):
            node.tag = node.tag.split("}", 1)[1]
    return root, version


def _text(node, path):
    value = node.findtext(path)
    return value.strip() if value and value.strip() else None


def _number(node, path, integer=False):
    text = _text(node, path)
    if text is None:
        return None
    value = int(text) if integer else float(text)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite CPHD parameter: {path}")
    return value


def extract_parameters(path, channel_id=None):
    """One explicitly supplied product applies to every pair in the current run."""
    root, version = read_cphd_xml(path)
    channels = root.findall("Channel/Parameters")
    available = [_text(node, "Identifier") for node in channels]
    if not available or None in available or len(set(available)) != len(available):
        raise ValueError("CPHD must contain uniquely identified channels")
    if channel_id is None:
        if len(channels) != 1:
            raise ValueError(f"Multiple CPHD channels; specify --cphd-channel from {available}")
        channel_id = available[0]
    if channel_id not in available:
        raise ValueError(f"Unknown CPHD channel {channel_id!r}; available: {available}")
    channel = channels[available.index(channel_id)]
    data_channel = next((node for node in root.findall("Data/Channel")
                         if _text(node, "Identifier") == channel_id), None)
    if data_channel is None:
        raise ValueError(f"Missing Data/Channel for {channel_id!r}")
    parameters = {
        "schema": "cphd_parameters_v1",
        "source_cphd": str(Path(path).resolve()),
        "cphd_version": version,
        "scope": "one_product_for_all_pairs_in_run",
        "collection": {
            "collector_name": _text(root, "CollectionID/CollectorName"),
            "core_name": _text(root, "CollectionID/CoreName"),
            "collect_type": _text(root, "CollectionID/CollectType"),
            "radar_mode": _text(root, "CollectionID/RadarMode/ModeType"),
            "collection_start_utc": _text(root, "Global/Timeline/CollectionStart"),
            "signal_domain": _text(root, "Global/DomainType"),
            "tx_time_min_s": _number(root, "Global/Timeline/TxTime1"),
            "tx_time_max_s": _number(root, "Global/Timeline/TxTime2"),
        },
        "channel": {
            "identifier": channel_id,
            "tx_polarization": _text(channel, "Polarization/TxPol"),
            "receive_polarization": _text(channel, "Polarization/RcvPol"),
            "saved_center_frequency_hz": _number(channel, "FxC"),
            "saved_bandwidth_hz": _number(channel, "FxBW"),
            "num_vectors": _number(data_channel, "NumVectors", integer=True),
            "num_samples_per_vector": _number(data_channel, "NumSamples", integer=True),
        },
        "reference_geometry": {
            "reference_channel_id": _text(root, "Channel/RefChId"),
        },
    }
    required = parameters["channel"]
    for name in ("tx_polarization", "receive_polarization", "saved_center_frequency_hz",
                 "saved_bandwidth_hz", "num_vectors", "num_samples_per_vector"):
        if required[name] is None:
            raise ValueError(f"Missing CPHD channel parameter: {name}")
    for name in ("saved_center_frequency_hz", "saved_bandwidth_hz",
                 "num_vectors", "num_samples_per_vector"):
        if required[name] <= 0:
            raise ValueError(f"CPHD channel parameter must be positive: {name}")
    geometry = root.find("ReferenceGeometry/Monostatic")
    if geometry is not None:
        parameters["reference_geometry"].update({
            "type": "monostatic",
            "side_of_track": _text(geometry, "SideOfTrack"),
            "slant_range_m": _number(geometry, "SlantRange"),
            "incidence_angle_deg": _number(geometry, "IncidenceAngle"),
            "graze_angle_deg": _number(geometry, "GrazeAngle"),
            "azimuth_angle_deg": _number(geometry, "AzimuthAngle"),
        })
    # Resolve waveform/receiver IDs for the selected channel only. Other
    # channels may use different configurations within the same product.
    for group, id_tag, xml_tag, fields in (
        ("transmit_waveforms", "TxWFId", "TxWFParameters", {
            "pulse_length_s": "PulseLength", "rf_bandwidth_hz": "RFBandwidth",
            "center_frequency_hz": "FreqCenter",
        }),
        ("receivers", "RcvId", "RcvParameters", {
            "sample_rate_hz": "SampleRate", "window_length_s": "WindowLength",
            "if_filter_bandwidth_hz": "IFFilterBW",
        }),
    ):
        parameters[group] = []
        for reference in channel.findall(f"TxRcv/{id_tag}"):
            identifier = reference.text.strip() if reference.text else ""
            node = next((node for node in root.findall(f"TxRcv/{xml_tag}")
                         if _text(node, "Identifier") == identifier), None)
            if node is None:
                raise ValueError(f"Unresolved CPHD {id_tag}: {identifier!r}")
            parameters[group].append({
                "identifier": identifier,
                **{name: _number(node, tag) for name, tag in fields.items()},
            })
    return parameters


def make_product_prompt(parameters):
    """Render acquisition context; geometry remains product-reference geometry."""
    collection = parameters["collection"]
    channel = parameters["channel"]
    parts = ["CPHD product acquisition parameters:"]
    for key, label in (("collector_name", "collector"), ("radar_mode", "radar mode"),
                       ("collect_type", "collection type"), ("signal_domain", "signal domain")):
        if collection[key] is not None:
            parts.append(f"{label}: {collection[key]};")
    parts.extend((
        f"transmit/receive polarization: {channel['tx_polarization']}/{channel['receive_polarization']};",
        f"saved-band center frequency: {channel['saved_center_frequency_hz'] / 1e9:.6f} GHz;",
        f"saved bandwidth: {channel['saved_bandwidth_hz'] / 1e6:.3f} MHz;",
        f"product signal array: {channel['num_vectors']} vectors, "
        f"{channel['num_samples_per_vector']} samples per vector;",
    ))
    for waveform in parameters["transmit_waveforms"]:
        for key, label, scale, unit in (
            ("pulse_length_s", "pulse length", 1e6, "us"),
            ("rf_bandwidth_hz", "transmitted RF bandwidth", 1e-6, "MHz"),
        ):
            if waveform[key] is not None:
                parts.append(f"waveform {waveform['identifier']} {label}: {waveform[key] * scale:.3f} {unit};")
    for receiver in parameters["receivers"]:
        if receiver["sample_rate_hz"] is not None:
            parts.append(f"receiver {receiver['identifier']} sample rate: "
                         f"{receiver['sample_rate_hz'] / 1e6:.3f} MHz;")
    geometry = parameters["reference_geometry"]
    if geometry.get("type") == "monostatic":
        reference = geometry["reference_channel_id"]
        parts.append(f"product reference geometry (channel {reference}):"
                     if reference is not None else "product reference geometry:")
        side = geometry["side_of_track"]
        if side in ("L", "R"):
            parts.append(f"{'left' if side == 'L' else 'right'}-looking;")
        for key, label, scale, unit in (
            ("incidence_angle_deg", "incidence angle", 1, "deg"),
            ("graze_angle_deg", "graze angle", 1, "deg"),
            ("azimuth_angle_deg", "azimuth angle", 1, "deg"),
            ("slant_range_m", "slant range", 1e-3, "km"),
        ):
            if geometry[key] is not None:
                parts.append(f"{label}: {geometry[key] * scale:.3f} {unit};")
    return " ".join(parts).rstrip(";") + "."
