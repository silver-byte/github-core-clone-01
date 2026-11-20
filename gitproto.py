"""
Git smart-HTTP wire protocol implementation.
Imports from gitcore and store.Repo.
"""

from typing import Generator, Tuple, List, Set, Optional, Dict, Any
from gitcore import (
    ObjectStore,
    read_pack,
    write_pack,
    parse_commit,
    commit_parents,
    is_ancestor,
    reachable_objects,
    closure_objects,
    PackError,
    ZERO_SHA,
    TYPE_NAME,
    TYPE_NUM,
)


# pkt-line constants
FLUSH = b"0000"
DELIM = b"0001"


def pkt(data: bytes) -> bytes:
    """Encode data as pkt-line with 4-byte hex length prefix."""
    if len(data) > 65516:
        raise ValueError("pkt-line data too long")
    length = len(data) + 4
    return f"{length:04x}".encode() + data


def pkt_text(s: str) -> bytes:
    """Encode string as pkt-line."""
    return pkt(s.encode())


def parse_packets(data: bytes, start: int = 0) -> Generator[Tuple[str, bytes, int], None, None]:
    """Parse pkt-lines from data. Yields (kind, payload, next_pos)."""
    pos = start
    n = len(data)
    while pos < n:
        if pos + 4 > n:
            break
        try:
            length = int(data[pos:pos+4], 16)
        except ValueError:
            break
        pos += 4
        if length == 0:
            yield ("flush", b"", pos)
            continue
        if length == 1:
            yield ("delim", b"", pos)
            continue
        if length < 4:
            # treat as flush
            yield ("flush", b"", pos)
            continue
        payload_len = length - 4
        if pos + payload_len > n:
            break
        payload = data[pos:pos+payload_len]
        pos += payload_len
        yield ("data", payload, pos)


def _read_pkt_line(data: bytes, pos: int) -> Tuple[bytes, int]:
    """Read a single pkt-line, return (payload, new_pos)."""
    if pos + 4 > len(data):
        return b"", pos
    try:
        length = int(data[pos:pos+4], 16)
    except ValueError:
        return b"", pos
    pos += 4
    if length == 0:
        return b"", pos
    if length == 1:
        return DELIM, pos
    if length < 4:
        return b"", pos
    payload_len = length - 4
    if pos + payload_len > len(data):
        return b"", pos
    payload = data[pos:pos+payload_len]
    pos += payload_len
    return payload, pos


def advertise(repo: Optional[Any], service: str, version2: bool) -> bytes:
    """Generate advertisement for upload-pack or receive-pack."""
    if repo is None:
        return b""
    
    out = bytearray()
    
    if version2:
        out.extend(pkt_text("version 2\n"))
        out.extend(pkt_text("agent=forge/1.0\n"))
        if service == "git-upload-pack":
            out.extend(pkt_text("ls-refs=unborn symrefs peel\n"))
            out.extend(pkt_text("fetch=shallow wait-for-done\n"))
            out.extend(pkt_text("object-format=sha1\n"))
        elif service == "git-receive-pack":
            out.extend(pkt_text("object-format=sha1\n"))
        out.extend(FLUSH)
        return bytes(out)
    
# version 0
    out.extend(pkt_text(f"# service={service}\n"))
    out.extend(FLUSH)
    
    store = repo.objects
    refs = dict(repo.refs)
    
    # Default branch for symref
    default_branch = repo.default_branch if hasattr(repo, 'default_branch') else "main"
    symref_cap = f"symref=HEAD:refs/heads/{default_branch}"
    
    if service == "git-upload-pack":
        caps = f"multi_ack thin-pack side-band side-band-64k ofs-delta deepen-since deepen-not {symref_cap}"
    else:
        caps = f"report-status side-band-64k delete-refs {symref_cap}"
    
    if not refs:
        # Empty repo: ONLY the capabilities^{} line
        out.extend(pkt_text(f"{ZERO_SHA} capabilities^{{\}}\0{caps}\n"))
    else:
        # Non-empty repo: HEAD line with caps, then refs
        head_target = repo.head_ref
        head_oid = refs.get(head_target, ZERO_SHA)
        if head_oid == ZERO_SHA:
            head_oid = ZERO_SHA
        
        first_line = f"{head_oid} HEAD\0{caps}\n"
        out.extend(pkt_text(first_line))
        
        # All refs sorted
        for refname in sorted(refs.keys()):
            oid = refs[refname]
            out.extend(pkt_text(f"{oid} {refname}\n"))
    
    out.extend(FLUSH)
    return bytes(out)


def _peel_tag(store: ObjectStore, sha: str) -> Optional[str]:
    """Peel a tag object to its commit target."""
    if not store.has(sha):
        return None
    obj_type, data = store.get(sha)
    if obj_type != "tag":
        return sha
    # Parse tag object
    lines = data.decode("utf-8", errors="replace").split("\n")
    for line in lines:
        if line.startswith("object "):
            obj_sha = line[7:]
            return _peel_tag(store, obj_sha)
    return None


def ls_refs(repo: Any, body: bytes) -> bytes:
    """Handle ls-refs request (v2)."""
    out = bytearray()
    
    # Parse request
    want_symrefs = False
    want_peel = False
    want_unborn = False
    ref_prefixes: List[str] = []
    
    state = "caps"
    pos = 0
    for kind, payload, pos in parse_packets(body, 0):
        if state == "caps":
            if kind == "delim":
                state = "args"
                continue
            # ignore capability lines
            continue
        if kind == "flush":
            break
        if kind != "data":
            continue
        line = payload.decode("utf-8", errors="replace").strip()
        if line.startswith("ref-prefix "):
            ref_prefixes.append(line[11:])
        elif line == "symrefs":
            want_symrefs = True
        elif line == "peel":
            want_peel = True
        elif line == "unborn":
            want_unborn = True
    
    store = repo.objects
    refs = dict(repo.refs)
    
    # HEAD line first if symrefs or unborn requested
    if want_symrefs or want_unborn:
        head_oid = refs.get(repo.head_ref, ZERO_SHA)
        if head_oid == ZERO_SHA:
            if want_symrefs:
                line = f"unborn HEAD symref-target:{repo.head_ref}"
            else:
                line = "unborn HEAD"
        else:
            line = f"{head_oid} HEAD"
            if want_symrefs:
                line += f" symref-target:{repo.head_ref}"
        out.extend(pkt_text(line + "\n"))
    
    # Refs sorted by name
    for refname in sorted(refs.keys()):
        # Filter by prefix
        if ref_prefixes:
            if not any(refname.startswith(p) for p in ref_prefixes):
                continue
        oid = refs[refname]
        line = f"{oid} {refname}"
        if want_peel:
            peeled = _peel_tag(store, oid)
            if peeled and peeled != oid:
                line += f" peeled:{peeled}"
        out.extend(pkt_text(line + "\n"))
    
    out.extend(FLUSH)
    return bytes(out)


def _collect_ancestors(store: ObjectStore, tips: List[str]) -> Set[str]:
    """Collect all ancestor commits of tips (inclusive)."""
    closure: Set[str] = set()
    stack: List[str] = list(tips)
    while stack:
        sha = stack.pop()
        if sha in closure:
            continue
        if not store.has(sha):
            continue
        obj_type, _ = store.get(sha)
        if obj_type != "commit":
            continue
        closure.add(sha)
        for p in commit_parents(store, sha):
            if p not in closure:
                stack.append(p)
    return closure


def _fetch_plan(
    repo: Any,
    wants: List[str],
    haves: List[str],
    shallow: List[str],
    deepen: Optional[int],
    deepen_not: List[str],
) -> Tuple[List[str], Set[str], Set[str]]:
    """Compute fetch plan: objects to send, new boundary, unshallow."""
    store = repo.objects
    
    # Have closure (commits the client has)
    have_closure = _collect_ancestors(store, [h for h in haves if store.has(h)])
    
    # Excluded commits (deepen-not)
    excluded = have_closure | _collect_ancestors(store, [d for d in deepen_not if store.has(d)])
    
    # Handle tag wants: if a want is a tag, include the tag and peel to commit
    want_commits: List[str] = []
    want_tags: List[str] = []
    for w in wants:
        if not store.has(w):
            continue
        obj_type, _ = store.get(w)
        if obj_type == "tag":
            want_tags.append(w)
            # Peel to commit
            peeled = _peel_tag(store, w)
            if peeled:
                want_commits.append(peeled)
        elif obj_type == "commit":
            want_commits.append(w)
    
    # BFS from wants by generation
    included: Dict[str, int] = {}  # sha -> gen
    queue: List[Tuple[str, int]] = [(w, 1) for w in want_commits]
    
    while queue:
        sha, gen = queue.pop(0)
        if sha in included:
            continue
        if sha in excluded:
            continue
        if not store.has(sha):
            continue
        obj_type, _ = store.get(sha)
        if obj_type != "commit":
            continue
        
        included[sha] = gen
        
        if deepen is not None and gen >= deepen:
            continue
        
        for p in commit_parents(store, sha):
            if p not in included and p not in excluded:
                queue.append((p, gen + 1))
    
    # Hold commits: haves and shallow tips that exist
    hold_commits = set(h for h in haves if store.has(h)) | set(s for s in shallow if store.has(s))
    
    # Objects to send: closure of included commits minus closure of hold commits minus closure of excluded
    included_tips = set(included.keys())
    send_objects_set = closure_objects(store, included_tips)
    
    hold_objects = closure_objects(store, hold_commits)
    excluded_objects = closure_objects(store, excluded)
    
    send_objects_set -= hold_objects
    send_objects_set -= excluded_objects
    
    # Add want tags themselves
    for t in want_tags:
        send_objects_set.add(t)
    
    send_objects = sorted(send_objects_set)
    
    # New boundary (only when deepen is not None)
    new_boundary: Set[str] = set()
    unshallow: Set[str] = set()
    
    if deepen is not None:
        # Commits at exactly deepen with parent outside included/excluded
        for sha, gen in included.items():
            if gen != deepen:
                continue
            parents = commit_parents(store, sha)
            has_outside_parent = False
            for p in parents:
                if p not in included and p not in excluded and store.has(p):
                    has_outside_parent = True
                    break
            if has_outside_parent:
                new_boundary.add(sha)
        
        # Old shallow commits that were not visited nor excluded (stay boundary)
        for s in shallow:
            if store.has(s) and s not in included and s not in excluded:
                new_boundary.add(s)
        
        # Unshallow: old shallow commits that are in included with gen < deepen
        for s in shallow:
            if s in included and included[s] < deepen:
                unshallow.add(s)
    
    return send_objects, new_boundary, unshallow


def upload_pack_v2(repo: Any, body: bytes) -> bytes:
    """Handle fetch request (v2)."""
    out = bytearray()
    
    # Parse request
    wants: List[str] = []
    haves: List[str] = []
    shallow: List[str] = []
    deepen: Optional[int] = None
    deepen_not: List[str] = []
    
    state = "command"
    pos = 0
    for kind, payload, pos in parse_packets(body, 0):
        if state == "command":
            if kind != "data":
                continue
            line = payload.decode("utf-8", errors="replace").strip()
            if line == "command=fetch":
                state = "caps"
            elif line == "command=ls-refs":
                return ls_refs(repo, body)
            continue
        if state == "caps":
            if kind == "delim":
                state = "args"
                continue
            # ignore capability lines
            continue
        if state == "args":
            if kind == "flush":
                break
            if kind != "data":
                continue
            line = payload.decode("utf-8", errors="replace").strip()
            if line.startswith("want "):
                wants.append(line[5:].split()[0])
            elif line.startswith("have "):
                haves.append(line[5:].split()[0])
            elif line.startswith("shallow "):
                shallow.append(line[8:].split()[0])
            elif line.startswith("deepen "):
                try:
                    deepen = int(line[7:].split()[0])
                except ValueError:
                    pass
            elif line.startswith("deepen-not "):
                deepen_not.append(line[11:].split()[0])
            # ignore other flags: done, thin-pack, no-progress, ofs-delta
    
    # Filter out ZERO_SHA and unknown objects
    effective_wants = [w for w in wants if w != ZERO_SHA and repo.objects.has(w)]
    if not effective_wants:
        return FLUSH
    
    send_objects, new_boundary, unshallow = _fetch_plan(repo, effective_wants, haves, shallow, deepen, deepen_not)
    
    # Build pack
    pack = None
    if send_objects or not haves:
        pack = write_pack(repo.objects, send_objects)
    
    sections: List[bytes] = []
    
    # Acknowledgments
    if haves:
        ack_data = bytearray()
        ack_data.extend(pkt_text("acknowledgments\n"))
        common = [h for h in haves if repo.objects.has(h)]
        if common:
            ack_data.extend(pkt_text(f"ACK {common[0]}\n"))
        else:
            ack_data.extend(pkt_text("NAK\n"))
        sections.append(bytes(ack_data))
    
    # Shallow info
    if new_boundary or unshallow:
        si_data = bytearray()
        si_data.extend(pkt_text("shallow-info\n"))
        for sha in sorted(new_boundary):
            si_data.extend(pkt_text(f"shallow {sha}\n"))
        for sha in sorted(unshallow):
            si_data.extend(pkt_text(f"unshallow {sha}\n"))
        sections.append(bytes(si_data))
    
    # Packfile
    if pack is not None:
        pf_data = bytearray()
        pf_data.extend(pkt_text("packfile\n"))
        # Chunk pack into <= 65515 byte pieces
        for i in range(0, len(pack), 65515):
            chunk = pack[i:i+65515]
            pf_data.extend(pkt(b"\x01" + chunk))
        sections.append(bytes(pf_data))
    
    # Join sections with DELIM, end with FLUSH
    for i, section in enumerate(sections):
        out.extend(section)
        if i < len(sections) - 1:
            out.extend(DELIM)
    out.extend(FLUSH)
    
    return bytes(out)


def upload_pack_v0(repo: Any, body: bytes) -> bytes:
    """Handle fetch request (v0)."""
    out = bytearray()
    
    wants: List[str] = []
    haves: List[str] = []
    deepen: Optional[int] = None
    
    pos = 0
    while pos < len(body):
        payload, pos = _read_pkt_line(body, pos)
        if payload == FLUSH:
            break
        if payload == DELIM:
            continue
        if not payload:
            continue
        line = payload.decode("utf-8", errors="replace").strip()
        if line.startswith("want "):
            # Strip capabilities after NUL
            sha = line[5:].split("\0")[0].split()[0]
            wants.append(sha)
        elif line.startswith("have "):
            haves.append(line[5:].split()[0])
        elif line.startswith("deepen "):
            try:
                deepen = int(line[7:].split()[0])
            except ValueError:
                pass
        elif line == "done":
            break
    
    # Always send NAK first
    out.extend(pkt_text("NAK\n"))
    
    # Filter out ZERO_SHA and unknown objects
    effective_wants = [w for w in wants if w != ZERO_SHA and repo.objects.has(w)]
    if not effective_wants:
        out.extend(FLUSH)
        return bytes(out)
    
    send_objects, _, _ = _fetch_plan(repo, effective_wants, haves, [], deepen, [])
    
    pack = write_pack(repo.objects, send_objects)
    
    # Send pack in sideband chunks
    for i in range(0, len(pack), 65515):
        chunk = pack[i:i+65515]
        out.extend(pkt(b"\x01" + chunk))
    
    out.extend(FLUSH)
    return bytes(out)


def receive_pack(repo: Any, body: bytes) -> bytes:
    """Handle receive-pack request."""
    # Parse command packets using parse_packets which stops at FLUSH
    commands: List[Tuple[str, str, str]] = []  # (old, new, ref)
    pos = 0
    
    for kind, payload, pos in parse_packets(body, 0):
        if kind == "flush":
            break
        if kind == "delim":
            continue
        if kind != "data":
            continue
        line = payload.decode("utf-8", errors="replace").strip()
        # Strip capability part after NUL
        if "\0" in line:
            line = line.split("\0")[0]
        parts = line.split()
        if len(parts) != 3:
            continue
        old, new, ref = parts
        commands.append((old, new, ref))
    
    # Remaining body is pack data
    pack_data = body[pos:]
    
    # Try to read pack
    unpack_error = None
    if pack_data:
        try:
            read_pack(repo.objects, pack_data)
        except PackError as e:
            unpack_error = str(e)
    
    # Validate refname
    def valid_refname(ref: str) -> bool:
        if not ref.startswith("refs/"):
            return False
        if ".." in ref:
            return False
        if "//" in ref:
            return False
        if ref.startswith("/"):
            return False
        if ref.endswith("/") or ref.endswith(".lock"):
            return False
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in ref):
            return False
        return True
    
    # Process commands
    results: List[Tuple[str, str]] = []  # (status, message)
    
    if unpack_error:
        for old, new, ref in commands:
            results.append(("ng", f"{ref} unpack failed"))
    else:
        # First pass: validate all
        validations: List[Optional[str]] = []
        for old, new, ref in commands:
            if not valid_refname(ref):
                validations.append("invalid ref")
                continue
            if new == ZERO_SHA:  # delete
                if ref not in repo.refs:
                    validations.append("reference does not exist")
                else:
                    validations.append(None)
            elif old == ZERO_SHA:  # create
                if ref in repo.refs:
                    validations.append("reference already exists")
                elif not repo.objects.has(new):
                    validations.append("not our object")
                else:
                    validations.append(None)
            else:  # update
                current = repo.refs.get(ref)
                if current != old:
                    validations.append("stale info")
                elif not repo.objects.has(new):
                    validations.append("not our object")
                elif not is_ancestor(repo.objects, old, new):
                    validations.append("non-fast-forward")
                else:
                    validations.append(None)
        
        # Second pass: apply
        for (old, new, ref), err in zip(commands, validations):
            if err:
                results.append(("ng", f"{ref} {err}"))
            else:
                if new == ZERO_SHA:
                    del repo.refs[ref]
                else:
                    repo.refs[ref] = new
                results.append(("ok", ref))
    
    # Build inner response
    inner = bytearray()
    if unpack_error:
        inner.extend(pkt_text(f"unpack {unpack_error}\n"))
    else:
        inner.extend(pkt_text("unpack ok\n"))
    for status, msg in results:
        inner.extend(pkt_text(f"{status} {msg}\n"))
    inner.extend(FLUSH)
    
    # Chunk inner response into sideband
    outer = bytearray()
    inner_bytes = bytes(inner)
    for i in range(0, len(inner_bytes), 65515):
        chunk = inner_bytes[i:i+65515]
        outer.extend(pkt(b"\x01" + chunk))
    outer.extend(FLUSH)
    
    return bytes(outer)