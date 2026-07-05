def replace_t_with_space(documents):
    """
    Replaces tab characters with spaces in each document's page content.

    Args:
        documents: A list of LangChain Document objects.

    Returns:
        The same list of Document objects with tabs stripped from page_content.
    """
    for doc in documents:
        doc.page_content = doc.page_content.replace('\t', ' ')
    return documents
