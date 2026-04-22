Polars IO plugin for reading the Databento Binary Encoding (DBN) format.

```python
import polars as pl
import polars_dbn as pldbn

path = "path/to/file.dbn.zst"  # works with zstd compression
df: pl.LazyFrame = pldbn.scan_dbn(path)

...
```