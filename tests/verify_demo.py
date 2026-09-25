import re

def verify_demo():
    with open('demo/index.html', 'r', encoding='utf-8') as f:
        text = f.read()

    print('File length:', len(text))
    has_double_curly = ('{{' in text) or ('}}' in text)
    print('Double curlies present:', has_double_curly)
    assert not has_double_curly, "Found unescaped double curly braces in demo/index.html!"

    panels = re.findall(r'id=[\"\']pnl-([^\"\']+)[\"\']', text)
    print('Panels found (' + str(len(panels)) + '):', panels)
    
    tabs = re.findall(r'onclick=[\"\']sw\(\'([^\']+)\'\)[\"\']', text)
    print('Tabs found (' + str(len(tabs)) + '):', tabs)

    assert len(panels) == 11, f"Expected 11 panels, found {len(panels)}"
    assert len(tabs) == 11, f"Expected 11 tabs, found {len(tabs)}"
    assert panels == tabs, f"Panels and tabs do not match: {panels} != {tabs}"

    # Check for canvas IDs
    canvas_ids = re.findall(r'<canvas[^>]*id=[\"\']([^\"\']+)[\"\']', text)
    print('Canvas charts (' + str(len(canvas_ids)) + '):', canvas_ids)
    expected_canvases = ['c-res-line', 'c-res-tau', 'c-sim-dist', 'c-param-chart', 'c-seed-bar', 'c-ettm1-chart']
    for cid in expected_canvases:
        assert cid in canvas_ids, f"Missing canvas chart ID: {cid}"

    print("ALL VERIFICATION CHECKS PASSED!")

if __name__ == '__main__':
    verify_demo()
