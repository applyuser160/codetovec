from unittest.mock import patch, MagicMock
from src.codetovec import CodeToVec

@patch("src.codetovec.AutoTokenizer")
@patch("src.codetovec.AutoModel")
def test_execute(mock_model, mock_tokenizer):
    mock_model.from_pretrained.return_value.to.return_value = MagicMock()
    mock_tokenizer.from_pretrained.return_value = MagicMock()

    cv = CodeToVec()

    source = """
def test():
    '''doc'''
    # comment
    a = 1
    return a
"""
    clean_all = cv._cleanse_python_code(source, True, True, True)
    assert 'doc' not in clean_all
    assert 'comment' not in clean_all
    assert '\n\n' not in clean_all

    cv.tokenizer.return_value.to.return_value = MagicMock()

    try:
        cv.execute(source, remove_comments=True, remove_docstrings=True, remove_blank_lines=True)
    except Exception:
        pass

    cv.tokenizer.assert_called_with(clean_all, return_tensors="pt")

def test_cleanse_python_code():
    cv = CodeToVec.__new__(CodeToVec)

    source = """
def foo():
    '''
    docstring
    '''
    # comment
    a = 1 # inline

    return a
"""
    clean = cv._cleanse_python_code(source, remove_comments=True, remove_docstrings=True, remove_blank_lines=True)
    assert 'docstring' not in clean
    assert '# comment' not in clean
    assert '# inline' not in clean
    assert 'a = 1' in clean
    assert 'def foo():' in clean
