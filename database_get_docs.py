#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

SELECT attribute_id, question_id FROM test_attribute_question WHERE attribute_id IN (SELECT id FROM test_attribute WHERE test_id IN (10,10);
"""

from __future__ import division  # need for python2
from __future__ import print_function
from __future__ import absolute_import

import sys
import ast
import argparse
import math
import logging

from collections import namedtuple
import numpy as np
import pandas as pd
import operator # for min

import mysql
import mysql.connector
from mysql.connector import errorcode

sys.path.append('.') 
sys.path.append('..')
#from database import settings
import settings

logging.basicConfig(level=logging.DEBUG)

#--------------------------------------
# work with database mysql
def read_config_file(filename):

	with open(filename, 'r') as f:
		info = f.read()
	return ast.literal_eval(info)


CONFIG = read_config_file(settings.DATABASE_CONFIG_FILE)
CONFIG['raise_on_warnings'] = True
CONFIG['connection_timeout'] = 100000 #9000


#------------------

def exec_query(query):

	assert type(query) is str

	cnx = mysql.connector.connect(**CONFIG)
	cursor = cnx.cursor()

	try:			
		cursor.execute(query)  # create table
		rows = cursor.fetchall()
	except Exception as e:
		logging.error("ERROR: {0}".format(e))
		raise Exception("Can not exec query {0}".format(query))
	else:
		logging.info("Query was permormed: {0}".format(query))
	
	cursor.close()
	cnx.close()	

	return rows


def exec_query_list(query_list):

	if type(query_list) is str:
		query_list = [query_list]

	cnx = mysql.connector.connect(**CONFIG)
	cursor = cnx.cursor()

	for query in query_list:
		try:			
			cursor.execute(query)  # create table
		except Exception as e:
			logging.error("ERROR: {0}".format(e))
			raise Exception("Can not exec query {0}".format(query))
		else:
			logging.info("Query was permormed: {0}".format(query))
	
	cursor.close()
	cnx.close()		


def create_table_query(query):
	
	table_name = query.split()[2]
	conn = mysql.connector.connect(**CONFIG)
	curs = conn.cursor()
	try:			
		curs.execute(query)  # create table
	except Exception as e:
		logging.error("ERROR: {0}".format(e))
		raise Exception("Can not create table {0}".format(table_name))
	else:
		logging.info("The table {0} was success created.".format(table_name))
	curs.close()
	conn.close()	


def clean_table(table_name, truncate=False):

	cnx = mysql.connector.connect(**CONFIG)
	cursor = cnx.cursor()
	if truncate:
		cursor.execute("TRUNCATE TABLE {0}".format(table_name))
	else:
		cursor.execute("DELETE FROM {0}".format(table_name))		
	cnx.commit()		
	cursor.close()
	cnx.close()	


#======================== LOAD DATA FROM DB ==================

def load_docs_from_db():
	""" Returns docs.
	Returns
	-------
	docs : dict, where key is tuple = (name, text)
	"""
	cnx = mysql.connector.connect(**CONFIG)
	cursor = cnx.cursor()

	cursor.execute("SELECT id, name, text FROM sc_syndrome")
	rows = cursor.fetchall()	
	docs_dict = {x[0]: {'name':x[1], 'text':x[2]} for x in rows}

	docs = dict()
	docs['ids']   = [x[0] for x in rows]
	docs['names'] = [x[1] for x in rows]
	docs['texts'] = [x[2] for x in rows]
	docs['size'] = len(docs['ids'])

	cursor.close()
	cnx.close()

	return docs


if __name__ == '__main__':

	docs = load_docs_from_db()
	print('Number of docs:', docs['size'])
	for i in range(docs['size']):
		print(docs['ids'][i], ':', docs['names'][i])

	#for k in docs:
	#	print(k, ':', docs[k]['name'])
	
