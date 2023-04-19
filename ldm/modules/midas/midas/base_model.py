import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        device = torch.device("cuda")  if torch.cuda.is_available()  else  torch.device("cpu")        
        parameters = torch.load(path, map_location=device)

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
