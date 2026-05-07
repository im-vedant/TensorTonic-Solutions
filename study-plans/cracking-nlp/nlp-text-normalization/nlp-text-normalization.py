def text_normalize(text, operations):
    """
    Returns: str
    """
    result = text
    for op in operations:
        if op == "lowercase":
            result = result.lower()
        elif op == "remove_punctuation":
            result = re.sub(r'[^\w\s]', '', result)
        elif op == "remove_digits":
            result = re.sub(r'\d', '', result)
        elif op == "collapse_whitespace":
            result = re.sub(r'\s+', ' ', result)
        elif op == "strip":
            result = result.strip()
    print(result)
    return result
