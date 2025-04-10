import multiprocessing as mp
from common.setup import bert_setup_args, setup_logger
from common.train import train

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    args = bert_setup_args()
    logger = setup_logger(args, args.model_type)

    logger.info("------NEW RUN-----")
    logger.info("Training/evaluation parameters %s", args)

    try:
        train(args, logger)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
