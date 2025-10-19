from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict

Edge = Tuple[str, str]

@dataclass(frozen=True)
class ArgumentationFramework:
    '''
    Defifnition of a directed grpah for arguments and attack 
    A : The set of argument names.
    R : The set of attack edges (u, v) meaning "u attacks v".
    '''
    A: Tuple[str, ...]
    R: Tuple[Edge, ...]

    @staticmethod
    def from_lists(A: List[str], R: List[Edge]) -> "ArgumentationFramework":
        '''Build an framework from python list '''
        A_u = tuple(dict.fromkeys([a.strip() for a in A if a.strip()]))
        R_u = tuple(sorted(set((u.strip(), v.strip()) for (u, v) in R)))
        return ArgumentationFramework(A=A_u, R=R_u)

    def attackers_map(self) -> Dict[str, List[str]]:
        '''return a dico of the list of the attecher'''
        atk = {a: [] for a in self.A}
        for (u, v) in self.R:
            if v in atk:
                atk[v].append(u)
        return atk

def parse_nodes(text: str) -> List[str]:
    '''utils for parsing a list of nodes from free text'''
    sep = "," if "," in text else " "
    return [t.strip() for t in text.replace("\n", sep).split(sep) if t.strip()]

def parse_edges(text: str) -> List[Edge]:
    '''utils for parsing edges from text one edge per line'''
    R: List[Edge] = []
    for line in text.strip().splitlines():
        p = line.replace(",", " ").split()
        if len(p) == 2:
            R.append((p[0], p[1]))
    return R
