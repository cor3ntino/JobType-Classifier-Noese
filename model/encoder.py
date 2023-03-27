from sentence_transformers import SentenceTransformer

class Encoder:

    def __init__(self, device):

        self.device = device

        # Load multilingual Transformer; supports 50+ languages.
        # For performance, see here:
        # https://www.sbert.net/docs/pretrained_models.html#multi-lingual-models
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        # 512 tokens to get more context from Job Description
        self.model.max_seq_length = 512
    
    def encode(self, sentences):

        return self.model.encode(
            sentences,
            show_progress_bar=False,
            device=self.device
        )
