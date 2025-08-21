from zlib import crc32

def string_to_float_hash(s, encoding="utf-8"):
    """
    Generates a deterministic float hash between 0.0 and 1.0 from a string.
    """
    # Encode the string to bytes
    byte_string = s.encode(encoding)
    
    # Compute the CRC32 hash, ensuring it's treated as an unsigned integer
    hash_value = crc32(byte_string) & 0xffffffff
    
    # Normalize the hash value to a float between 0 and 1
    # Divide by 2**32 (the maximum value for a 32-bit unsigned integer)
    normalized_hash = float(hash_value) / (2**32)
    
    return normalized_hash