"""
Git core object/pack engine - pure Python 3.13 standard library.
"""

import hashlib
import zlib
import struct
from typing import Dict, List, Tuple, Set, Optional, Union, Iterator

# Constants
ZERO_SHA = "0" * 40
OBJ_COMMIT = 1
OBJ_TREE = 2
OBJ_BLOB = 3
OBJ_TAG = 4

TYPE_NAME = {
    OBJ_COMMIT: "commit",
    OBJ_TREE: "tree",
    OBJ_BLOB: "blob",
    OBJ_TAG: "tag",
}

TYPE_NUM = {v: k for k, v in TYPE_NAME.items()}


def hash_object(obj_type: str, data: bytes) -> str:
    """Compute SHA1 of Git object: type size\\0data."""
    header = f"{obj_type} {len(data)}".encode() + b"\0"
    return hashlib.sha1(header + data).hexdigest()


class ObjectStore:
    """In-memory object store mapping SHA -> (type_str, data_bytes)."""

    def __init__(self) -> None:
        self.objects: Dict[str, Tuple[str, bytes]] = {}

    def add(self, obj_type: str, data: bytes) -> str:
        """Store object, return its SHA."""
        sha = hash_object(obj_type, data)
        self.objects[sha] = (obj_type, data)
        return sha

    def add_raw(self, sha: str, obj_type: str, data: bytes) -> None:
        """Store object under known SHA; verify hash matches."""
        expected = hash_object(obj_type, data)
        if sha != expected:
            raise ValueError(f"SHA mismatch: expected {expected}, got {sha}")
        self.objects[sha] = (obj_type, data)

    def has(self, sha: str) -> bool:
        return sha in self.objects

    def get(self, sha: str) -> Tuple[str, bytes]:
        """Return (type, data) or raise KeyError."""
        return self.objects[sha]

    def get_data(self, sha: str) -> bytes:
        """Return data only or raise KeyError."""
        return self.objects[sha][1]

    def __len__(self) -> int:
        return len(self.objects)


# Commit helpers
def parse_commit(data: bytes) -> Dict[str, Union[str, List[str]]]:
    """Parse commit object into dict."""
    lines = data.decode("utf-8", errors="replace").split("\n")
    result: Dict[str, Union[str, List[str]]] = {
        "tree": "",
        "parents": [],
        "author": "",
        "committer": "",
        "message": "",
    }
    i = 0
    for i, line in enumerate(lines):
        if line == "":
            break
        if line.startswith("tree "):
            result["tree"] = line[5:]
        elif line.startswith("parent "):
            result["parents"].append(line[7:])
        elif line.startswith("author "):
            result["author"] = line[7:]
        elif line.startswith("committer "):
            result["committer"] = line[10:]
    result["message"] = "\n".join(lines[i + 1 :])
    return result


def build_commit(
    tree: str,
    parents: List[str],
    author: str,
    committer: str,
    message: str,
) -> bytes:
    """Build commit object from components."""
    lines = [f"tree {tree}"]
    for p in parents:
        lines.append(f"parent {p}")
    lines.append(f"author {author}")
    lines.append(f"committer {committer}")
    lines.append("")
    lines.append(message)
    return "\n".join(lines).encode("utf-8")


# Tree helpers
def parse_tree(data: bytes) -> List[Tuple[str, str, str]]:
    """Parse tree object into list of (mode, name, sha)."""
    entries = []
    i = 0
    n = len(data)
    while i < n:
        sp = data.find(b" ", i)
        mode = data[i:sp].decode("ascii")
        i = sp + 1
        nul = data.find(b"\0", i)
        name = data[i:nul].decode("utf-8", errors="replace")
        i = nul + 1
        sha = data[i:i+20].hex()
        i += 20
        entries.append((mode, name, sha))
    return entries


def build_tree(entries: List[Tuple[str, str, str]]) -> bytes:
    """Build tree object from entries; sort with git rules."""
    def sort_key(e: Tuple[str, str, str]) -> str:
        mode, name, _ = e
        return name + "/" if mode == "40000" else name
    
    entries = sorted(entries, key=sort_key)
    out = bytearray()
    for mode, name, sha in entries:
        if mode == "40000":
            mode = "40000"  # normalize tree mode
        out.extend(mode.encode("ascii"))
        out.append(0x20)  # space
        out.extend(name.encode("utf-8"))
        out.append(0x00)  # NUL
        out.extend(bytes.fromhex(sha))
    return bytes(out)


# Tag helpers
def parse_tag(data: bytes) -> Dict[str, str]:
    """Parse tag object into dict."""
    lines = data.decode("utf-8", errors="replace").split("\n")
    result = {"object": "", "type": "", "tag": "", "tagger": "", "message": ""}
    i = 0
    for i, line in enumerate(lines):
        if line == "":
            break
        if line.startswith("object "):
            result["object"] = line[7:]
        elif line.startswith("type "):
            result["type"] = line[5:]
        elif line.startswith("tag "):
            result["tag"] = line[4:]
        elif line.startswith("tagger "):
            result["tagger"] = line[7:]
    result["message"] = "\n".join(lines[i + 1 :])
    return result


def build_tag(
    obj_sha: str,
    obj_type: str,
    tag_name: str,
    tagger: str,
    message: str,
) -> bytes:
    """Build tag object from components."""
    lines = [
        f"object {obj_sha}",
        f"type {obj_type}",
        f"tag {tag_name}",
        f"tagger {tagger}",
        "",
        message,
    ]
    return "\n".join(lines).encode("utf-8")


# Pack reading
class PackError(Exception):
    """Raised for pack format errors."""
    pass


def _read_varint(data: bytes, offset: int) -> Tuple[int, int]:
    """Read varint from data at offset. Returns (value, new_offset)."""
    val = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise PackError("Unexpected EOF reading varint")
        b = data[offset]
        offset += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
        if shift > 63:
            raise PackError("Varint too long")
    return val, offset


def _read_ofs_delta_offset(data: bytes, offset: int) -> Tuple[int, int]:
    """Read OFS_DELTA base offset (git's special varint encoding).
    
    Encoding: 
      b = next byte; offset = b & 0x7f
      while b & 0x80:
          b = next byte
          offset = ((offset + 1) << 7) | (b & 0x7f)
    Returns (offset_value, new_offset).
    """
    if offset >= len(data):
        raise PackError("Unexpected EOF reading OFS_DELTA offset")
    b = data[offset]
    offset += 1
    val = b & 0x7F
    while b & 0x80:
        if offset >= len(data):
            raise PackError("Unexpected EOF reading OFS_DELTA offset continuation")
        b = data[offset]
        offset += 1
        val = ((val + 1) << 7) | (b & 0x7F)
    return val, offset


def _write_varint(val: int) -> bytes:
    """Write varint (for pack header)."""
    out = bytearray()
    while val >= 0x80:
        out.append((val & 0x7F) | 0x80)
        val >>= 7
    out.append(val)
    return bytes(out)


def _write_pack_header(obj_type: int, size: int) -> bytes:
    """Write pack object header: type in bits 4-6, size in lower bits + continuation."""
    # First byte: (type << 4) | (size & 0x0F), with MSB set if size > 15
    first = (obj_type << 4) | (size & 0x0F)
    size >>= 4
    out = bytearray()
    if size:
        out.append(first | 0x80)
        while size >= 0x80:
            out.append(size & 0x7F | 0x80)
            size >>= 7
        out.append(size)
    else:
        out.append(first)
    return bytes(out)


def apply_delta(base: bytes, delta: bytes) -> bytes:
    """Apply delta to base, return target bytes."""
    offset = 0
    src_size, offset = _read_varint(delta, offset)
    tgt_size, offset = _read_varint(delta, offset)
    
    if src_size != len(base):
        raise PackError(f"Delta source size {src_size} != base size {len(base)}")
    
    out = bytearray()
    while offset < len(delta):
        cmd = delta[offset]
        offset += 1
        if cmd & 0x80:  # copy from base
            # offset and size encoded in cmd and following bytes
            cp_offset = 0
            cp_size = 0
            if cmd & 0x01:
                cp_offset |= delta[offset] << 0
                offset += 1
            if cmd & 0x02:
                cp_offset |= delta[offset] << 8
                offset += 1
            if cmd & 0x04:
                cp_offset |= delta[offset] << 16
                offset += 1
            if cmd & 0x08:
                cp_offset |= delta[offset] << 24
                offset += 1
            if cmd & 0x10:
                cp_size |= delta[offset] << 0
                offset += 1
            if cmd & 0x20:
                cp_size |= delta[offset] << 8
                offset += 1
            if cmd & 0x40:
                cp_size |= delta[offset] << 16
                offset += 1
            if cp_size == 0:
                cp_size = 0x10000
            if cp_offset + cp_size > len(base):
                raise PackError("Delta copy out of bounds")
            out.extend(base[cp_offset:cp_offset + cp_size])
        elif cmd != 0:  # literal insert
            size = cmd
            if offset + size > len(delta):
                raise PackError("Delta literal overrun")
            out.extend(delta[offset:offset + size])
            offset += size
        else:
            raise PackError("Invalid delta opcode 0")
    
    if len(out) != tgt_size:
        raise PackError(f"Delta target size {len(out)} != expected {tgt_size}")
    return bytes(out)


def read_pack(store: ObjectStore, data: bytes) -> int:
    """Parse pack file, verify checksum, add objects to store. Returns count."""
    if len(data) < 12:
        raise PackError("Pack too small for header")
    
    # Verify header
    magic, version, count = struct.unpack(">4sII", data[:12])
    if magic != b"PACK":
        raise PackError("Invalid pack signature")
    if version != 2:
        raise PackError(f"Unsupported pack version {version}")
    
    offset = 12
    # Staging maps: offset -> (type_num, data), sha -> (type_str, data)
    staged_by_offset: Dict[int, Tuple[int, bytes]] = {}
    staged_by_sha: Dict[str, Tuple[str, bytes]] = {}
    
    for _ in range(count):
        if offset >= len(data) - 20:
            raise PackError("Unexpected EOF reading object header")
        
        obj_offset = offset
        # Read type/size varint
        first = data[offset]
        offset += 1
        obj_type = (first >> 4) & 0x07
        size = first & 0x0F
        shift = 4
        while first & 0x80:
            if offset >= len(data):
                raise PackError("Unexpected EOF in size varint")
            first = data[offset]
            offset += 1
            size |= (first & 0x7F) << shift
            shift += 7
        
        if obj_type in (OBJ_COMMIT, OBJ_TREE, OBJ_BLOB, OBJ_TAG):
            # Undeltified object: size is uncompressed size, data is zlib stream
            # Feed data incrementally until EOF, then use unused_data to find boundary
            decompressor = zlib.decompressobj()
            raw = b""
            while offset < len(data) - 20 and not decompressor.eof:
                chunk = data[offset:offset + 4096]
                offset += len(chunk)
                try:
                    raw += decompressor.decompress(chunk)
                except zlib.error as e:
                    raise PackError(f"Zlib decompress failed: {e}")
            if not decompressor.eof:
                raise PackError("Incomplete zlib stream")
            # Put back unused data
            unused = decompressor.unused_data
            if unused:
                offset -= len(unused)
            if len(raw) != size:
                raise PackError(f"Decompressed size {len(raw)} != header size {size}")
            type_str = TYPE_NAME.get(obj_type)
            if type_str is None:
                raise PackError(f"Unknown object type {obj_type}")
            sha = hash_object(type_str, raw)
            # Verify hash integrity
            if hash_object(type_str, raw) != sha:
                raise PackError("object hash mismatch")
            staged_by_offset[obj_offset] = (obj_type, raw)
            staged_by_sha[sha] = (type_str, raw)
            
        elif obj_type == 6:  # OFS_DELTA
            # Read OFS_DELTA base offset (git's special encoding)
            ofs_delta_offset, offset = _read_ofs_delta_offset(data, offset)
            base_offset = obj_offset - ofs_delta_offset
            if base_offset not in staged_by_offset:
                raise PackError(f"OFS_DELTA base at offset {base_offset} not found")
            base_type, base_data = staged_by_offset[base_offset]
            # Read delta data (compressed zlib stream, size is uncompressed delta size)
            decompressor = zlib.decompressobj()
            delta = b""
            while offset < len(data) - 20 and not decompressor.eof:
                chunk = data[offset:offset + 4096]
                offset += len(chunk)
                try:
                    delta += decompressor.decompress(chunk)
                except zlib.error as e:
                    raise PackError(f"Zlib decompress failed: {e}")
            if not decompressor.eof:
                raise PackError("Incomplete zlib stream for delta")
            unused = decompressor.unused_data
            if unused:
                offset -= len(unused)
            if len(delta) != size:
                raise PackError(f"Delta decompressed size {len(delta)} != header size {size}")
            try:
                result = apply_delta(base_data, delta)
            except PackError as e:
                raise PackError(f"Delta apply failed: {e}")
            type_str = TYPE_NAME[base_type]
            sha = hash_object(type_str, result)
            # Verify hash integrity
            if hash_object(type_str, result) != sha:
                raise PackError("object hash mismatch")
            staged_by_offset[obj_offset] = (base_type, result)
            staged_by_sha[sha] = (type_str, result)
            
        elif obj_type == 7:  # REF_DELTA
            if offset + 20 > len(data) - 20:
                raise PackError("REF_DELTA base SHA overrun")
            base_sha = data[offset:offset + 20].hex()
            offset += 20
            # Read delta data (compressed zlib stream, size is uncompressed delta size)
            decompressor = zlib.decompressobj()
            delta = b""
            while offset < len(data) - 20 and not decompressor.eof:
                chunk = data[offset:offset + 4096]
                offset += len(chunk)
                try:
                    delta += decompressor.decompress(chunk)
                except zlib.error as e:
                    raise PackError(f"Zlib decompress failed: {e}")
            if not decompressor.eof:
                raise PackError("Incomplete zlib stream for delta")
            unused = decompressor.unused_data
            if unused:
                offset -= len(unused)
            if len(delta) != size:
                raise PackError(f"Delta decompressed size {len(delta)} != header size {size}")
            # Resolve base
            if base_sha in staged_by_sha:
                base_type_str, base_data = staged_by_sha[base_sha]
            elif store.has(base_sha):
                base_type_str, base_data = store.get(base_sha)
            else:
                raise PackError(f"REF_DELTA base {base_sha} not found")
            base_type = TYPE_NUM[base_type_str]
            try:
                result = apply_delta(base_data, delta)
            except PackError as e:
                raise PackError(f"Delta apply failed: {e}")
            type_str = TYPE_NAME[base_type]
            sha = hash_object(type_str, result)
            # Verify hash integrity
            if hash_object(type_str, result) != sha:
                raise PackError("object hash mismatch")
            staged_by_offset[obj_offset] = (base_type, result)
            staged_by_sha[sha] = (type_str, result)
            
        else:
            raise PackError(f"Unknown object type {obj_type}")
    
    # Verify trailing SHA1
    if offset + 20 != len(data):
        raise PackError(f"Trailing checksum at wrong position: {offset} vs {len(data)}")
    trailer = data[offset:offset + 20]
    expected = hashlib.sha1(data[:offset]).digest()
    if trailer != expected:
        raise PackError("Pack checksum mismatch")
    
    # Commit staged objects to store
    for sha, (type_str, raw) in staged_by_sha.items():
        if not store.has(sha):
            store.add_raw(sha, type_str, raw)
    
    return len(staged_by_sha)


# Pack writing
def write_pack(store: ObjectStore, shas: List[str]) -> bytes:
    """Write pack file for given SHAs (no deltas)."""
    out = bytearray()
    out.extend(b"PACK")
    out.extend(struct.pack(">II", 2, len(shas)))  # version 2, count
    
    for sha in shas:
        obj_type, data = store.get(sha)
        type_num = TYPE_NUM[obj_type]
        header = _write_pack_header(type_num, len(data))
        compressed = zlib.compress(data)
        out.extend(header)
        out.extend(compressed)
    
    # Trailer SHA1
    trailer = hashlib.sha1(out).digest()
    out.extend(trailer)
    return bytes(out)


# Reachability and ancestry
def commit_parents(store: ObjectStore, sha: str) -> List[str]:
    """Return list of parent SHAs for commit, empty if not a commit."""
    if not store.has(sha):
        return []
    obj_type, data = store.get(sha)
    if obj_type != "commit":
        return []
    parsed = parse_commit(data)
    return list(parsed["parents"])


def is_ancestor(store: ObjectStore, anc: str, desc: str) -> bool:
    """True if anc == desc or anc reachable from desc by walking parents."""
    if anc == desc:
        return True
    visited: Set[str] = set()
    queue: List[str] = [desc]
    while queue:
        cur = queue.pop()
        if cur in visited:
            continue
        visited.add(cur)
        if cur == anc:
            return True
        for p in commit_parents(store, cur):
            if p not in visited:
                queue.append(p)
    return False


def _compute_exclude_closure(store: ObjectStore, exclude: Set[str]) -> Set[str]:
    """Compute full commit closure of exclude tips."""
    closure: Set[str] = set()
    stack: List[str] = list(exclude)
    while stack:
        sha = stack.pop()
        if sha in closure:
            continue
        closure.add(sha)
        for p in commit_parents(store, sha):
            if p not in closure:
                stack.append(p)
    return closure


def _compute_full_object_closure(store: ObjectStore, tips: Set[str]) -> Set[str]:
    """Compute full object closure (commits, trees, blobs, tags) from commit tips."""
    closure: Set[str] = set()
    stack: List[str] = list(tips)
    while stack:
        sha = stack.pop()
        if sha in closure:
            continue
        closure.add(sha)
        if not store.has(sha):
            continue
        obj_type, data = store.get(sha)
        if obj_type == "commit":
            parsed = parse_commit(data)
            for p in parsed["parents"]:
                if p not in closure:
                    stack.append(p)
            tree = parsed["tree"]
            if tree not in closure:
                stack.append(tree)
        elif obj_type == "tree":
            for _, _, entry_sha in parse_tree(data):
                if entry_sha not in closure:
                    stack.append(entry_sha)
        elif obj_type == "tag":
            parsed = parse_tag(data)
            obj = parsed["object"]
            if obj not in closure:
                stack.append(obj)
    return closure


def reachable_objects(
    store: ObjectStore,
    tips: List[str],
    exclude: Set[str] = frozenset(),
    max_depth: Optional[int] = None,
    held: Set[str] = frozenset(),
) -> Tuple[Set[str], Set[str]]:
    """
    Walk commits from tips by generation.
    Returns (objects_to_send, new_shallow_boundary).
    """
    # Precompute exclude closure (commits only)
    exclude_closure = _compute_exclude_closure(store, exclude)
    
    # Precompute full object closure of exclude tips
    exclude_obj_closure = _compute_full_object_closure(store, exclude_closure)
    
    # Also include held commits in exclude closure
    held_closure = _compute_exclude_closure(store, held)
    exclude_closure |= held_closure
    exclude_obj_closure |= _compute_full_object_closure(store, held_closure)
    
    # BFS by generation from tips
    included_commits: Set[str] = set()
    boundary: Set[str] = set()
    visited: Set[str] = set()
    
    # Queue of (sha, generation)
    queue: List[Tuple[str, int]] = [(t, 1) for t in tips]
    
    while queue:
        sha, gen = queue.pop(0)
        if sha in visited:
            continue
        visited.add(sha)
        
        if sha in exclude_closure:
            continue
        
        if max_depth is not None and gen > max_depth:
            continue
        
        # This commit is included
        included_commits.add(sha)
        
        parents = commit_parents(store, sha)
        has_parent_outside = False
        for p in parents:
            if p not in included_commits and p not in exclude_closure:
                if max_depth is None or gen < max_depth:
                    queue.append((p, gen + 1))
                if max_depth is not None and gen == max_depth:
                    has_parent_outside = True
            elif p not in included_commits:
                # Parent is excluded or missing
                if max_depth is not None and gen == max_depth:
                    has_parent_outside = True
        
        if max_depth is not None and gen == max_depth and has_parent_outside:
            boundary.add(sha)
    
    # Now collect all objects reachable from included commits
    included_objects = _compute_full_object_closure(store, included_commits)
    
    # Subtract objects reachable from excluded commits
    objects_to_send = included_objects - exclude_obj_closure
    
    return objects_to_send, boundary


def closure_objects(store: ObjectStore, tips: Set[str]) -> Set[str]:
    """Full object closure from commit tips."""
    return _compute_full_object_closure(store, tips)


# Tree manipulation
def tree_lookup(store: ObjectStore, tree_sha: str, path: str) -> Optional[Tuple[str, str]]:
    """Walk path components from tree_sha, return (mode, sha) or None."""
    if not path or path == "/":
        return None
    parts = [p for p in path.split("/") if p]
    cur_sha = tree_sha
    for part in parts:
        if not store.has(cur_sha):
            return None
        obj_type, data = store.get(cur_sha)
        if obj_type != "tree":
            return None
        entries = parse_tree(data)
        found = False
        for mode, name, sha in entries:
            if name == part:
                cur_sha = sha
                found = True
                break
        if not found:
            return None
    if not store.has(cur_sha):
        return None
    obj_type, _ = store.get(cur_sha)
    # Return mode of the last entry - we need to look it up again
    # Actually, we should return the mode of the final entry
    # Let's re-walk to get the mode
    if not parts:
        return None
    # Re-walk to get mode
    cur_sha = tree_sha
    last_mode = ""
    for part in parts:
        obj_type, data = store.get(cur_sha)
        entries = parse_tree(data)
        for mode, name, sha in entries:
            if name == part:
                last_mode = mode
                cur_sha = sha
                break
    return (last_mode, cur_sha)


def tree_list(store: ObjectStore, tree_sha: str, path: str) -> Optional[List[Tuple[str, str, str]]]:
    """List entries of directory at path. None if missing or not a tree."""
    if not store.has(tree_sha):
        return None
    obj_type, data = store.get(tree_sha)
    if obj_type != "tree":
        return None
    
    if not path or path == "/":
        return parse_tree(data)
    
    parts = [p for p in path.split("/") if p]
    cur_sha = tree_sha
    for part in parts:
        entries = parse_tree(store.get_data(cur_sha))
        found = False
        for mode, name, sha in entries:
            if name == part:
                cur_sha = sha
                found = True
                break
        if not found:
            return None
    
    if not store.has(cur_sha):
        return None
    obj_type, data = store.get(cur_sha)
    if obj_type != "tree":
        return None
    return parse_tree(data)


def _tree_entries_to_dict(entries: List[Tuple[str, str, str]]) -> Dict[str, Tuple[str, str]]:
    """Convert tree entries to dict name -> (mode, sha)."""
    return {name: (mode, sha) for mode, name, sha in entries}


def _dict_to_tree_entries(d: Dict[str, Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    """Convert dict to sorted tree entries."""
    return [(mode, name, sha) for name, (mode, sha) in d.items()]


def tree_update(
    store: ObjectStore,
    tree_sha: Optional[str],
    path: str,
    blob_sha: Optional[str],
) -> str:
    """Set or delete file at path, return new root tree SHA."""
    if not path or path == "/":
        raise ValueError("Cannot update root tree directly")
    
    parts = [p for p in path.split("/") if p]
    if not parts:
        raise ValueError("Empty path")
    
    def _update(entries: Dict[str, Tuple[str, str]], parts: List[str], blob_sha: Optional[str]) -> str:
        """Recursively update tree entries. Returns new tree SHA."""
        if len(parts) == 1:
            # Leaf level: set or delete the entry
            name = parts[0]
            if blob_sha is None:
                # Delete
                if name in entries:
                    del entries[name]
            else:
                # Set file
                entries[name] = ("100644", blob_sha)
        else:
            # Intermediate level: recurse into subtree
            name = parts[0]
            sub_sha = None
            if name in entries:
                mode, sha = entries[name]
                if mode == "40000":
                    sub_sha = sha
            
            # Get sub-entries
            sub_entries: Dict[str, Tuple[str, str]] = {}
            if sub_sha and store.has(sub_sha):
                obj_type, data = store.get(sub_sha)
                if obj_type == "tree":
                    sub_entries = _tree_entries_to_dict(parse_tree(store.get_data(sub_sha)))
            
            # Recurse
            new_sub_sha = _update(sub_entries, parts[1:], blob_sha)
            
            # Handle result
            if new_sub_sha == ZERO_SHA or (blob_sha is None and new_sub_sha == _empty_tree_sha(store)):
                # Subtree became empty - remove the entry
                if name in entries:
                    del entries[name]
            else:
                entries[name] = ("40000", new_sub_sha)
        
        # Build and return new tree SHA
        tree_entries = [(mode, name, sha) for name, (mode, sha) in entries.items()]
        return store.add("tree", build_tree(tree_entries))
    
    # Get initial entries
    entries: Dict[str, Tuple[str, str]] = {}
    if tree_sha and store.has(tree_sha):
        obj_type, data = store.get(tree_sha)
        if obj_type == "tree":
            entries = _tree_entries_to_dict(parse_tree(data))
    
    return _update(entries, parts, blob_sha)


def _empty_tree_sha(store: ObjectStore) -> str:
    """Get the SHA of an empty tree."""
    return store.add("tree", build_tree([]))


def tree_from_flat(store: ObjectStore, files: Dict[str, str]) -> str:
    """Build nested trees from {path: blob_sha}."""
    # Build tree structure
    root: Dict[str, Union[Dict, str]] = {}
    for path, blob_sha in files.items():
        parts = [p for p in path.split("/") if p]
        cur = root
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                cur[part] = blob_sha
            else:
                if part not in cur:
                    cur[part] = {}
                elif isinstance(cur[part], str):
                    # Conflict: file where directory expected
                    cur[part] = {}
                cur = cur[part]
    
    def build_tree_recursive(node: Dict[str, Union[Dict, str]]) -> str:
        entries = []
        for name, val in sorted(node.items()):
            if isinstance(val, str):
                entries.append(("100644", name, val))
            else:
                sha = build_tree_recursive(val)
                entries.append(("40000", name, sha))
        return store.add("tree", build_tree(entries))
    
    return build_tree_recursive(root)


def flatten_tree(store: ObjectStore, tree_sha: str, prefix: str = "") -> Dict[str, Tuple[str, str]]:
    """Return all blob entries recursively as {path: (mode, sha)}."""
    result: Dict[str, Tuple[str, str]] = {}
    if not store.has(tree_sha):
        return result
    obj_type, data = store.get(tree_sha)
    if obj_type != "tree":
        return result
    for mode, name, sha in parse_tree(data):
        full_path = prefix + name
        if mode == "40000":
            result.update(flatten_tree(store, sha, full_path + "/"))
        else:
            result[full_path] = (mode, sha)
    return result


def diff_trees(
    store: ObjectStore,
    old_tree: Optional[str],
    new_tree: Optional[str],
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """Compare flattened maps; None tree = empty."""
    old_map = flatten_tree(store, old_tree) if old_tree else {}
    new_map = flatten_tree(store, new_tree) if new_tree else {}
    
    all_paths = set(old_map.keys()) | set(new_map.keys())
    result = {}
    for path in sorted(all_paths):
        old_mode_sha = old_map.get(path)
        new_mode_sha = new_map.get(path)
        old_sha = old_mode_sha[1] if old_mode_sha else None
        new_sha = new_mode_sha[1] if new_mode_sha else None
        result[path] = (old_sha, new_sha)
    return result