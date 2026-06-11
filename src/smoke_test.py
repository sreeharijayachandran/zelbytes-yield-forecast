import pandas as pd
import numpy as np
data = np.array(
    [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
    dtype=[("a", "i4"), ("b", "i8"), ("c", "f8")],
)
df3 = pd.DataFrame(data, columns=["a", "b", "c"])
print(df3)
print(df3.pop("a"))
print(df3)