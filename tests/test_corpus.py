def test_source_units_meet_minimum_without_losing_content():
    from starter.src.corpus import source_units
    text = '\n\n'.join(chr(65 + i) * 1000 for i in range(10))
    units = source_units(text)
    assert all(len(unit['text']) >= 1500 for unit in units)
    assert all(text[u['start']:u['end']] == u['text'] for u in units)
    assert ''.join(u['text'] for u in units).replace('\n', '') == text.replace('\n', '')


def test_html_excludes_navigation_and_preserves_code():
    from starter.src.corpus import extract_html
    html = '<html><nav>ignore</nav><div role="main"><h1>Title</h1><p>A rule.</p><pre>print(1)</pre></div></html>'
    assert extract_html(html.encode()) == 'Title\n\nA rule.\n\nprint(1)'


def test_html_keeps_python_code_indentation():
    from starter.src.corpus import extract_html
    html = '<div role="main"><pre>def f():\n    if True:\n        return 1\n</pre></div>'
    assert extract_html(html.encode()) == 'def f():\n    if True:\n        return 1'
