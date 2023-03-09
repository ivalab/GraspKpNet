from .dbctdet import DbCtdetDebugger
from .trictdet import TriCtdetDebugger

debugger_factory = {
    "trictdet": TriCtdetDebugger,
    "dbctdet": DbCtdetDebugger,
}
