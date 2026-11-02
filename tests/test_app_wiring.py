import os
import shutil
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["QT_QPA_FONTDIR"] = r"C:\Windows\Fonts"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "app"))

from PySide6.QtWidgets import QApplication          # noqa: E402

import main as M                                    # noqa: E402

app = QApplication([])
w = M.Main()

checks = []
checks.append(("window title", w.windowTitle().startswith("FlyMorph v")))
checks.append(("four tasks", len([k for k, v in M.TASKS.items() if v[2]]) == 4))
checks.append(("admin label", w.lbl_out.text().startswith("输出位置：")))

# every enabled task resolves to an existing model in ../models
for label, (key, model_file, enabled, folder) in M.TASKS.items():
    if not enabled:
        continue
    path = M._find_model(model_file)
    checks.append(("model %s" % key, os.path.exists(path)))
    checks.append(("folder %s" % key, folder == key))

ok = True
for name, passed in checks:
    print(("ok   " if passed else "FAIL ") + name)
    ok = ok and passed
print("ALL PASS" if ok else "FAILURES PRESENT")
