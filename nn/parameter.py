import torch

class Parameter(torch.Tensor):
    """
    Tensor used as a module parameter.

    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient.
    """

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.empty(0)
        if type(data) is torch.Tensor or type(data) is Parameter:
            return torch.Tensor._make_subclass(cls, data, requires_grad)
        
        t = data.detatch().requires_grad_(requires_grad)
        if type(t) is not type(data):
            raise RuntimeError(
                f"Creating a parameter from an instance of type {type(data).__name__} "
                f"requires that detatch() returns an instance of the same type, but return "
                f"type {type(t).__name__} was found instead."
            )
        
        t._is_param = True
        return t
    
    def _deepcopy_(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self) (
                self.data.clone(
                    memory_format=torch.preserve_format
                ),
                self.requires_grad
            )
            memo[id(self)] = result
            return result
        
    def __repr__(self) -> str:
        return "Parameter containing:\n" + super().__repr__()
    
class Buffer(torch.Tensor):
    """
    Tensor used within a module that is not consider a parameter.

    Args:
        data (Tensor): buffer tensor.
        persistent (bool, optional): whether the buffer is a part of the module's
            :attr:'state_dict'. Default: ''True''
    """

    def __new__(cls, data=None, *, persistent=True):
        if data is None:
            data = torch.empty(0)

        t = data.detach().requires_grad_(data.requires_grad)
        t.persistent = persistent
        t._is_buffer = True

        return t
    
    __torch_function__ = torch._C._disabled_torch_function_impl