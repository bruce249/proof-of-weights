import inspect
import toploc
from toploc.poly import build_proofs

print("build_proofs", inspect.signature(build_proofs))
print(inspect.getsource(build_proofs))
