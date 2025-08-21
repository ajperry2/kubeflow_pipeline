from zlib import crc32

def string_to_float_hash(s, encoding="utf-8"):
    """
    Generates a float hash between 0 and 1 from a string.
    """
    byte_string = s.encode(encoding)
    hash_value = crc32(byte_string) & 0xffffffff
    normalized_hash = float(hash_value) / (2**32)
    return normalized_hash