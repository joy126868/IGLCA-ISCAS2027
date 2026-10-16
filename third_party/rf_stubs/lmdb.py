"""Import-only stand-in for the lmdb package.

third_party/Retinexformer/basicsr/utils/lmdb_util.py imports lmdb at module level (it is only
used by the LMDB-creation utilities).  LOLv1 training reads PNGs from disk (io_backend: disk)
and never opens an LMDB, so any real use of this stub is an error.
"""


def open(*args, **kwargs):  # noqa: A001 - mirrors lmdb.open
    raise RuntimeError("lmdb stub: LMDB backends are not used in these runs")
