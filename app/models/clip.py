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
        # we normalize the text features to have a unit norm, 
        # which is a common practice in CLIP-based models to ensure that the similarity scores are computed correctly.
        self._text_feat=feat
    
    @torch.no_grad
    def score_pill(self,img:Image.Image)-> float:

        if self._text_feat is None:
            raise RuntimeError("Call set_prompt (prompt) before scoring images.")
        
        x=self.preprocess(img).unsqueeze(0).to(self.device)
        # why we unsequeeze the image tensor?
        # The unsqueeze(0) operation is used to add an extra dimension to the image tensor, 
        # which is necessary because the CLIP model expects a batch of images as input.
        # By unsqueezing the image tensor, we create a batch of size 1, 
        # allowing us to pass a single image through the model for feature extraction and similarity scoring.
        feat=self.model.encode_image(x)
        feat=feat/feat.norm(dim=1,keepdim=True)

        sim=(feat @self._text_feat.T).item()
        return sim 
