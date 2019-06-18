# Launch multiple gloo processes through qsub. Benchmark the running times 
# of ring-based and grid-based (w/ and w/o failures) all-reduce algorithms.
# The number of elements ranges from 1e5 to 1e8
# Usage:
#   ./python3 gloo_qsub.py [#_processes] [#_node_requested] [groups]
#

from pathlib import Path
import subprocess
import sys

PBS_HEADER = '#!/bin/bash\n#PBS -l select=1:ncpus=60\n\n'


size = int(sys.argv[1])
node = int(sys.argv[2])
group = int(sys.argv[3])
nps = size // node

subprocess.run('rm -f ~/tmp/*', shell=True)
subprocess.run('rm -f ~/gloo_n*', shell=True)
subprocess.run(f'mkdir -p ~/output_{size}_{group}', shell=True)

p = Path(Path.home(), 'gloo_n.sh')
script = PBS_HEADER

script += 'sleep 15\n'

p.write_text(script)
p.chmod(0o755)

subprocess.run('qsub ./{}'.format(p.name), shell=True)

elements = [[1 * 10**i, 2 * 10**i, 5 * 10**i] for i in range(5, 9)]
elements = [e for el in elements for e in el]

for n in range(node):
    p = Path(Path.home(), 'gloo_n{}.sh'.format(n))
    script = PBS_HEADER

    for i in range(n*nps, (n+1)*nps):
        script += f'rm -f ~/output_{size}_{group}/{i}.txt\n'
        script += f'cat $PBS_NODEFILE > ~/output_{size}_{group}/{i}.txt\n'

        for element in elements:
            if i == (n+1)*nps - 1:
                script += f'PREFIX=test{size}{group}_{element} ELEMENT={element} ' \
                    f'SIZE={size} RANK={i} GROUP={group} ~/gloo/build/gloo/examples/example2' \
                    f' >> ~/output_{size}_{group}/{i}.txt\n\n'
            else:
                script += f'PREFIX=test{size}{group}_{element} ELEMENT={element} ' \
                    f'SIZE={size} RANK={i} GROUP={group} ~/gloo/build/gloo/examples/example2' \
                    f' >> ~/output_{size}_{group}/{i}.txt &\n\n'

    p.write_text(script)
    p.chmod(0o755)

    subprocess.run('qsub ./{}'.format(p.name), shell=True)
