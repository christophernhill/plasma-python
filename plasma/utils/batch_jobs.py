from pprint import pprint
import yaml
import datetime
import uuid
import sys,os,getpass
import subprocess as sp
import numpy as np

def generate_working_dirname(run_directory):
    s = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    s += "_{}".format(uuid.uuid4())
    return run_directory + s



def get_executable_name(conf):
    shallow = conf['model']['shallow']
    if shallow:
        executable_name = conf['paths']['shallow_executable']
        use_mpi = False
    else:
        executable_name = conf['paths']['executable']
        use_mpi = True
    return executable_name,use_mpi


def start_slurm_job(subdir,num_nodes,i,conf,shallow):
    executable_name,use_mpi = get_executable_name(conf)
    os.system(" ".join(["cp -p",executable_name,subdir]))
    script = create_slurm_script(subdir,num_nodes,i,executable_name,use_mpi)
    sp.Popen("sbatch "+script,shell=True)

def create_slurm_script(subdir,num_nodes,idx,executable_name,use_mpi):
    filename = "run_{}_nodes.cmd".format(num_nodes)
    filepath = subdir+filename
    user = getpass.getuser()
    sbatch_header = create_sbatch_header(num_nodes,use_mpi,idx)
    with open(filepath,"w") as f:
        for line in sbatch_header:
            f.write(line)
        f.write('module load anaconda\n')
        f.write('source activate frnn\n')
        f.write('module load cudatoolkit/8.0 cudnn/cuda-8.0/6.0 openmpi/cuda-8.0/intel-17.0/2.1.0/64 intel/17.0/64/17.0.4.196 intel-mkl/2017.3/4/64\n')
        # f.write('rm -f /tigress/{}/model_checkpoints/*.h5\n'.format(user))
        f.write('cd {}\n'.format(subdir))
        f.write('export OMPI_MCA_btl=\"tcp,self,sm\"\n')
        f.write('srun python {}\n'.format(executable_name))
        f.write('echo "done."')

    return filepath

def create_sbatch_header(num_nodes,use_mpi,idx):
    if not use_mpi:
        assert(num_nodes == 1)
    lines = []
    lines.append('#!/bin/bash\n')
    lines.append('#SBATCH -t 06:00:00\n')
    lines.append('#SBATCH -N '+str(num_nodes)+'\n')
    if use_mpi:
        lines.append('#SBATCH --ntasks-per-node=4\n')
        lines.append('#SBATCH --ntasks-per-socket=2\n')
    else:
        lines.append('#SBATCH --ntasks-per-node=1\n')
        lines.append('#SBATCH --ntasks-per-socket=1\n')
    lines.append('#SBATCH --gres=gpu:4\n')
    lines.append('#SBATCH -c 4\n')
    lines.append('#SBATCH --mem-per-cpu=0\n')
    lines.append('#SBATCH -o {}.out\n'.format(idx))
    lines.append('\n\n')
    return lines

def copy_files_to_environment(subdir):
    from plasma.conf import conf
    normalization_dir = os.path.dirname(conf['paths']['normalizer_path'])
    if os.path.isdir(normalization_dir):
        print("Copying normalization to")
        os.system(" ".join(["cp -rp",normalization_dir,os.path.join(subdir,os.path.basename(normalization_dir))]))
