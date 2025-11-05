from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings


def get_embedding_model(args):
    if args.retrieval_model == "openai":
        return OpenAIEmbeddings()
    elif args.retrieval_model in args.retrieval_model_path_dict.keys():
        return HuggingFaceEmbeddings(model_name=args.retrieval_model_path_dict[args.retrieval_model])
    else:
        raise NotImplementedError

    