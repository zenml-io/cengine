from typing import Dict, Text, Any, List, Union
from cengine import Split, Index, Categorize


def no_split() -> Split:
    """ Helper function to create a no-split configuration
    """
    return Split()


def index_split(by: Text,
                ratio: Dict[Text, Any] = None) -> Split:
    """ Helper function to create a indexed split configuration

    :param by: name of the column to index by
    :param ratio: dict to define the splits, e.g. {"train":0.8, "eval":0.2}

    :return: an instance of Split with the proper configuration
    """
    return Split(index=Index(by=by, ratio=ratio))


def category_split(by: Text,
                   ratio: Dict[Text, Any] = None,
                   categories: Union[Dict, List] = None) -> Split:
    """ Helper function to create a categorical split configuration

    :param by: name of the column to categorize by
    :param ratio: dict to define the splits, e.g. {"train":0.8, "eval":0.2}
    :param categories: dict or list to define the categories

    :return: an instance of Split with the proper configuration
    """
    return Split(categorize=Categorize(by=by,
                                       ratio=ratio,
                                       categories=categories))


def hybrid_split(index_by: Text,
                 category_by: Text,
                 index_ratio: Dict[Text, Any] = None,
                 category_ratio: Dict[Text, Any] = None) -> Split:
    """ Helper function to create a hybrid split configuration with both index
    and categorical elements

    :param index_by: name of the column to index by
    :param category_by: name of the column to categorize by
    :param index_ratio: dict to define the index split ratios
    :param category_ratio: dict to define the category split ratios

    :return: an instance of Split with the proper configuration
    """
    return Split(index=Index(by=index_by, ratio=index_ratio),
                 categorize=Categorize(by=category_by, ratio=category_ratio))
