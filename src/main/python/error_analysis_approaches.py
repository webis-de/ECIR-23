from trectools import TrecQrel, TrecRun
import pandas as pd


def unjudged_documents_before_judged_documents():
    run = create_run([
        rl('du-1', 1),
        rl('du-2', 2),
        rl('du-3', 3),
        rl('du-4', 4),
        rl('du-5', 5),

        rl('d-1', 6),
        rl('d-2', 7),
        rl('d-3', 8),
        rl('d-4', 9),
        rl('d-5', 10),
    ])

    qrels_incomplete = create_qrels([
        ql('d-1', 1),
        ql('d-2', 1),
        ql('d-3', 1),
        ql('d-4', 1),
        ql('d-5', 1),
    ])

    qrels_complete = create_qrels([
        ql('d-1', 1),
        ql('d-2', 1),
        ql('d-3', 1),
        ql('d-4', 1),
        ql('d-5', 1),

        ql('du-1', 0),
        ql('du-2', 0),
        ql('du-3', 0),
        ql('du-4', 0),
        ql('du-5', 0),
    ])

    return {
        'run': run,
        'qrels_incomplete': qrels_incomplete,
        'qrels_complete': qrels_complete,
        'description': 'Condensed Lists can be gamed by inserting unjudged irrelevant documents before relevant ones'
    }


def good_run_unjudged_pos_5_easy_topic():
    run = create_run([
        rl('r-1', 1),
        rl('r-2', 2),
        rl('r-3', 3),
        rl('nr-1', 4),
        rl('u-1', 5),
    ])

    qrels = create_qrels([
        ql('r-1', 1),
        ql('r-2', 1),
        ql('r-3', 1),
        ql('r-4', 1),
        ql('r-5', 1),
        ql('r-6', 1),
        ql('r-7', 1),
        ql('r-8', 1),

        ql('nr-1', 0),
        ql('nr-2', 0),
        ql('nr-3', 0),
    ])

    return {
        'run': run,
        'qrels': qrels,
    }


def good_run_unjudged_pos_5_difficult_topic():
    run = create_run([
        rl('r-1', 1),
        rl('r-2', 2),
        rl('r-3', 3),
        rl('nr-1', 4),
        rl('u-1', 5),
    ])

    qrels = create_qrels([
        ql('r-1', 1),
        ql('r-2', 1),
        ql('r-3', 1),
        ql('r-4', 1),

        ql('nr-1', 0),
        ql('nr-2', 0),
        ql('nr-3', 0),
        ql('nr-4', 0),
        ql('nr-5', 0),
        ql('nr-6', 0),
        ql('nr-7', 0),
        ql('nr-8', 0),
        ql('nr-9', 0),
    ])

    return {
        'run': run,
        'qrels': qrels,
    }


def bad_run_unjudged_pos_5_easy_topic():
    run = create_run([
        rl('nr-1', 1),
        rl('nr-2', 2),
        rl('nr-3', 3),
        rl('r-1', 4),
        rl('u-1', 5),
    ])

    qrels = create_qrels([
        ql('r-1', 1),
        ql('r-2', 1),
        ql('r-3', 1),
        ql('r-4', 1),
        ql('r-5', 1),
        ql('r-6', 1),
        ql('r-7', 1),
        ql('r-8', 1),

        ql('nr-1', 0),
        ql('nr-2', 0),
        ql('nr-3', 0),
    ])

    return {
        'run': run,
        'qrels': qrels,
    }


def bad_run_unjudged_pos_5_difficult_topic():
    run = create_run([
        rl('nr-1', 1),
        rl('nr-2', 2),
        rl('nr-3', 3),
        rl('r-1', 4),
        rl('u-1', 5),
    ])

    qrels = create_qrels([
        ql('r-1', 1),
        ql('r-2', 1),
        ql('r-3', 1),
        ql('r-4', 1),

        ql('nr-1', 0),
        ql('nr-2', 0),
        ql('nr-3', 0),
        ql('nr-4', 0),
        ql('nr-5', 0),
        ql('nr-6', 0),
        ql('nr-7', 0),
        ql('nr-8', 0),
        ql('nr-9', 0),
    ])

    return {
        'run': run,
        'qrels': qrels,
    }


def create_run(entries):
    ret = TrecRun()
    ret.run_data = pd.DataFrame(entries)

    return ret


def create_qrels(entries):
    ret = TrecQrel()
    ret.qrels_data = pd.DataFrame(entries)

    return ret


def ql(doc_id, rel):
    return {'query': '1', 'q0': 0, 'docid': doc_id, 'rel': rel}


def rl(doc_id, pos, score=None):
    score = 3000-pos if not score else score
    return {'query': '1', 'q0': 'Q0', 'docid': doc_id, 'rank': pos, 'score': score, 'system': 'a'}


def as_latex(data):
    return ''
