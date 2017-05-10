import os
import re

import requests


def parse_application_id(url):
    m = re.search('application_[0-9]+_[0-9]+', url)
    return m.group(0)


def fetch_mapping(typename):
    return requests.get(
        'http://es.development.s2.dev.ai2:9200/%s/%s/_mapping' %
        (typename, typename)
    ).json()


class S2Pipeline(object):
    def __init__(self, application_id=None):
        if application_id:
            self.application_id = application_id
        else:
            self.set_application_id_from_dev()

    def set_application_id_from_dev(self):
        mapping = fetch_mapping('paper')
        self.application_id = parse_application_id(
            mapping['paper']['mappings']['paper']['_meta']['source']
        )

    def get_base_url(self):
        return "s3a://ai2-s2-pipeline/output/publish/" + self.application_id + '/'

    def get_data_url(self, data_type):
        return self.get_base_url() + data_type + '/'

    def get_authors_s3(self):
        return self.get_data_url('BuildEsAuthors')

    def get_figures_s3(self):
        return self.get_data_url('BuildEsFigures')

    def get_author_suggest_s3(self):
        return self.get_data_url('BuildEsAuthorsSuggest')

    def get_datasets_suggest_s3(self):
        return self.get_data_url('BuildEsDataSetsSuggest')

    def get_citations_s3(self):
        return self.get_data_url('BuildEsCitations')

    def get_paper_social_links_s3(self):
        return self.get_data_url('BuildEsPaperSocialLinks')

    def get_papers_s3(self):
        return self.get_data_url('BuildEsPapers')


def add_jar(jar):
    os.environ["PYSPARK_SUBMIT_ARGS"] = "--jars " + ",".join([jar]
                                                            ) + " pyspark-shell"


def add_package(package):
    os.environ["PYSPARK_SUBMIT_ARGS"
              ] = "--packages " + ",".join([package]) + " pyspark-shell"


def init_spark(spark_home):
    os.environ['SPARK_HOME'] = '/data/spark-2.0.0-bin-hadoop2.7/'

    import findspark
    findspark.add_packages('org.apache.hadoop:hadoop-aws:2.7.3')
    os.environ['PYSPARK_SUBMIT_ARGS'
              ] = ' --driver-memory 31g ' + os.environ['PYSPARK_SUBMIT_ARGS']
    findspark.init()


def get_local_context(local_dir='/tmp'):
    import pyspark
    conf = pyspark.SparkConf() \
        .setMaster('local[32]') \
        .setAppName('notebook') \
        .set('spark.local.dir', local_dir)

    sc = pyspark.SparkContext(conf=conf)
    return sc


def get_remote_context():
    import pyspark
    conf = pyspark.SparkConf()

    conf.set("spark.executor.memory", "31g")
    conf.set("spark.core.connection.ack.wait.timeout", "1200")

    # create the context
    sc = pyspark.SparkContext(conf=conf)
    return sc
