
import re

def normalize_gene_id(g: str) -> str:
    """Normalize gene/locus identifiers.

    - Strips whitespace
    - Uppercases
    - Removes isoform suffix after '.' (e.g., AT1G01010.1 -> AT1G01010)
    """
    if g is None:
        return ""
    g = str(g).strip().upper()
    g = re.sub(r"\s+", "", g)
    g = g.split(".")[0]
    return g
