function [] = reload_python_workspace(env)
%env = 1: Local machine
% env = 2: Queen's remote machine
%Unload Module
%clear classes
mcmf_path = [];
if (env == 1)
    mcmf_path = 'C:\Ibrahim-Workspace-Main\Research\Queen''s University\PhD\Research\Dr. Sharief & Dr. Hossam\Repos\PBTA.Simulation\Greedy.MCMF';
elseif (env == 2)
    mcmf_path = 'C:\Users\ibrahim.amer\Ibrahim-Workspace\Repos\Task.Replication\PBTA.Simulation\Greedy.MCMF';
end
module_name = 'MCMF';
if count(py.sys.path, mcmf_path) == 0
    insert(py.sys.path,int32(0), mcmf_path);
end
mod = py.importlib.import_module(module_name);
py.importlib.reload(mod);

end



