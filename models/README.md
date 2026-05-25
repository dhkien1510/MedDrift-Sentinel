# Models

## Encoder Selection & Evaluation

MedDrift-Sentinel **does not train or fine-tune** any models from scratch. Instead, the system leverages a wide range of **pre-trained encoder models** from HuggingFace to extract embeddings for drift detection. The project has been systematically tested across multiple encoder families to find the best combinations for medical VQA drift monitoring.

### Image Encoders Tested

| Model | Architecture | Domain |
|---|---|---|
| `microsoft/rad-dino-maira-2` | DINOv2 | Radiology-specific |
| `microsoft/rad-dino` | DINOv2 | Radiology-specific |
| `facebook/dinov2-base` | DINOv2 | General vision |
| `google/vit-base-patch16-224` | ViT | General vision |

### Text Encoders Tested

| Model | Architecture | Domain |
|---|---|---|
| `NeuML/pubmedbert-base-embeddings` | SentenceBERT | Biomedical NLP |
| `NeuML/pubmedbert-base-embeddings-8M` | Model2Vec | Biomedical NLP |
| `NeuML/pubmedbert-base-embeddings-matryoshka` | SentenceBERT | Biomedical NLP |
| `dmis-lab/biobert-v1.1` | BERT | Biomedical NLP |
| `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` | BERT | Biomedical NLP |
| `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | BERT | Biomedical NLP |
| `pritamdeka/S-PubMedBert-MS-MARCO` | SentenceBERT | Biomedical NLP |

### Why No Local Checkpoints?

The drift detection architecture relies on **pre-computed reference embeddings** generated from these pre-trained encoders — not on model weights stored locally. The embeddings are stored as `.npy` files under `data/reference_data/` and `data/drift_scenarios/`, organized by encoder name.

To regenerate embeddings with different encoders or settings, use the scripts in `meddrift-ai-service/scripts/`:
- `build_reference.py` — Generate reference embeddings for all configured encoders
- `build_drift_data.py` — Generate drift scenario embeddings at multiple severity levels
- `build_multimodal_reference.py` — Build joint PCA fusion for multimodal drift detection

The active encoder can be changed at runtime via `configs/drift_config.yaml` or through the web dashboard's configuration panel.
