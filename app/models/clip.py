import torch 
import open_clip
from PIL import Image

class ClipScapper:

    def __init__(self,model_name="ViT-B-32",pretrained="openai",device=None):
        self.device=device or("cuda" if torch.cuda.is_available() else "cpu")
        self.model,_,self.preprocess=open_clip.create_model_and_transforms(
            model_name,pretrained=pretrained
        )
        self.model=self.model.to(self.device).eval()
        self.tokenizer=open_clip.get_tokenizer(model_name)
        self._text_feat=None

    @torch.no_grad
    def set_prompt(self,prompt:str):
        text=self.tokenizer([prompt]).to(self.device)
        feat=self.model.encode_text(text)
        feat=feat/feat.norm(dim=1,keepdim=True)
        self._text_feat=feat
    
    @torch.no_grad
    def score_pill(self,img:Image.Image)-> float:

        if self._text_feat is None:
            raise RuntimeError("Call set_prompt (prompt) before scoring images.")
        
        x=self.preprocess(img).unsqueeze(0).to(self.device)
        feat=self.model.encode_image(x)
        feat=feat/feat.norm(dim=1,keepdim=True)

        sim=(feat @self._text_feat.T).item()
        return sim 
