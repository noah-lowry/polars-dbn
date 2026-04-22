### Polars IO plugin for reading the Databento Binary Encoding (DBN) format

#### Installation
```sh
uv add git+https://github.com/noah-lowry/polars-dbn
```
#### Usage

```python
import polars as pl
import polars_dbn as pldbn

path = "path/to/file.dbn.zst"  # works with zstd compression
df: pl.LazyFrame = pldbn.scan_dbn(path)

...
```