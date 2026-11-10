def text_to_bits(text, message_bits=96):
    """Pack UTF-8 text into a fixed-size payload, padding trailing bits."""
    raw = text.encode("utf-8")
    capacity = message_bits // 8 - 1
    if message_bits < 8 or capacity < 0:
        raise ValueError("message_bits must be at least 8")
    if len(raw) > min(capacity, 255):
        raise ValueError("Text exceeds {} UTF-8 bytes".format(min(capacity, 255)))
    payload = bytes([len(raw)]) + raw + bytes(capacity - len(raw))
    bits = [(byte >> shift) & 1 for byte in payload for shift in range(7, -1, -1)]
    return bits + [0] * (message_bits - len(bits))


def bits_to_text(bits):
    if len(bits) < 8:
        raise ValueError("At least 8 bits are required for a text payload")
    # Ignore the final incomplete byte, reserved as padding for non-byte-aligned models.
    byte_aligned_bits = len(bits) - len(bits) % 8
    values = []
    for start in range(0, byte_aligned_bits, 8):
        value = 0
        for bit in bits[start:start + 8]:
            value = (value << 1) | int(bit)
        values.append(value)
    if not values or values[0] > len(values) - 1:
        raise ValueError("Invalid text payload length")
    return bytes(values[1:1 + values[0]]).decode("utf-8")
