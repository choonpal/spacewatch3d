"""Offline boundary around original producers; only explicit I/O/class binding."""
import ast
import hashlib
import json
import runpy
import socket
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def denied(*args, **kwargs):
    raise PermissionError("Upstream worker is offline; required model files must already exist")


def raw_namespace(source, base, vocabulary):
    """Keep mask arithmetic; bind original fixed person16 to current class ID."""
    source = Path(source)
    text = source.read_text()
    catalog = json.loads(Path(vocabulary).read_text())
    person = [r['id'] for r in catalog if r['name'] == 'person']
    floor = [r['id'] for r in catalog if r['name'] == 'floor']
    ceiling = [r['id'] for r in catalog if r['name'] == 'ceiling']
    if len(person) != 1 or floor != [1] or ceiling != [2]:
        raise ValueError('Original raw producer requires one person class and floor1/ceiling2')
    tree = ast.parse(text)
    changes = []
    class BindPerson(ast.NodeTransformer):
        def visit_Compare(self, node):
            if (len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq) and
                    isinstance(node.left, ast.Subscript) and isinstance(node.left.value, ast.Name) and node.left.value.id == 'z' and
                    len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value == 16):
                changes.append({'line': node.lineno, 'old_person_id': 16, 'bound_person_id': person[0]})
                node.comparators[0] = ast.copy_location(ast.Constant(person[0]), node.comparators[0])
            return node
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'masks':
            BindPerson().visit(node)
    if len(changes) != 2:
        raise ValueError('Original raw person-mask source changed; expected exactly two class comparisons')
    ast.fix_missing_locations(tree)
    namespace = {'__name__': 'bound_original_raw_producer', '__file__': str(source)}
    exec(compile(tree, str(source), 'exec'), namespace)
    namespace['BASE'] = Path(base)
    binding = {'original_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'derived_ast_sha256': hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
               'derived_source_sha256': hashlib.sha256((ast.unparse(tree) if hasattr(ast,'unparse') else ast.dump(tree,include_attributes=False)).encode()).hexdigest(),
               'derived_source_representation': 'unparsed_python' if hasattr(ast,'unparse') else 'ast_dump',
               'class_binding_only': changes, 'base_io_override': str(base),
               'mask_algorithm_parameters_changed': False, 'vocabulary_sha256': hashlib.sha256(Path(vocabulary).read_bytes()).hexdigest()}
    return namespace, binding


def main():
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    mode = sys.argv[1]
    if mode == 'stage':
        job_path = Path(sys.argv[2]).resolve()
        job = json.loads(job_path.read_text())
        if job['action'] not in ('inputs', 'infer', 'reconstruct', 'mesh', 'semantic', 'scene'):
            raise ValueError('Only accepted upstream actions are allowed; no extra semantic rescue, virtual planning/Gaussian/API stages')
        sys.argv = ['spacewatch_pipeline.worker', str(job_path)]
        runpy.run_module('spacewatch_pipeline.worker', run_name='__main__')
    elif mode == 'raw':
        job_path = Path(sys.argv[2]).resolve()
        job = json.loads(job_path.read_text())
        ns, binding = raw_namespace(job['original_raw_source'], job['raw_base'], job['vocabulary'])
        action = job['raw_action']; cid = job['capture_id']
        if action == 'extract': ns['extract'](cid)
        elif action == 'masks': ns['masks'](cid, job['predictions'])
        elif action == 'coordinates': ns['coordinates'](cid, job['source_run'], job['predictions'])
        else: raise ValueError('Unknown original raw action')
        (job_path.parent / 'raw_binding_receipt.json').write_text(json.dumps(binding, indent=2)+'\n')
    elif mode == 'pano':
        source = Path(sys.argv[2]).resolve(); sys.argv = [str(source)] + sys.argv[3:]
        sys.path.insert(0, str(source.parent)); runpy.run_path(str(source), run_name='__main__')
    elif mode == 'profile':
        source = Path(sys.argv[2]).resolve(); sys.argv = [str(source)] + sys.argv[3:]
        runpy.run_path(str(source), run_name='__main__')
    else: raise ValueError('Unknown offline producer mode')


if __name__ == '__main__': main()
