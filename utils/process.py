from transformers.data.processors.squad import *


def _create_examples(input_data, set_type):
    is_training = set_type == "train"
    examples = []
    for entry in tqdm(input_data):
        # print(entry.keys())
        if "title" in entry:
            title = entry["title"]
        else:
            title = "no_title"
        for paragraph in entry["paragraphs"]:
            context_text = paragraph["context"]
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position_character = None
                answer_text = None
                answers = []

                if "is_impossible" in qa:
                    is_impossible = qa["is_impossible"]
                else:
                    is_impossible = False

                if not is_impossible:
                    if is_training:
                        answer = qa["answers"][0]
                        answer_text = answer["text"]
                        start_position_character = answer["answer_start"]
                    else:
                        answers = qa["answers"]

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    title=title,
                    is_impossible=is_impossible,
                    answers=answers,
                )

                examples.append(example)
    return examples


class SquadProcessor:
    """
    Processor for the SQuAD data set.
    Overriden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and version 2.0 of SQuAD, respectively.
    """

    def __init__(self, train_file=None, dev_file=None):
        self.train_file = train_file
        self.dev_file = dev_file

    def get_train_examples(self):

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
                self.train_file, "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return _create_examples(input_data, "train")

    def get_dev_examples(self):

        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
                self.dev_file, "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return _create_examples(input_data, "dev")


def load_and_cache_examples(args, path, tokenizer, is_training):
    cached_filename = "{}".format(str(args.max_seq_length))
    cached_prefix = os.path.join("cached", os.path.splitext(list(os.path.split(path))[-1])[0]).replace(".", "_")
    cached_path = os.path.join(cached_prefix, cached_filename)
    if os.path.exists(cached_path):
        dataset, examples, features = torch.load(cached_path)
    else:
        if is_training:
            processor = SquadProcessor(train_file=path)
            examples = processor.get_train_examples()
        else:
            processor = SquadProcessor(dev_file=path)
            examples = processor.get_dev_examples()
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=is_training,
            return_dataset="pt",
            threads=1,
        )
        if not os.path.exists(cached_prefix):
            os.makedirs(cached_prefix)
        torch.save((dataset, examples, features), cached_path)
    return dataset, examples, features
