#!/usr/bin/env python

import boto
import boto.dynamodb2

import time
import json
import random



KINESIS_STREAM_NAME = 'random_int'
DYNAMO_TABLE_NAME = 'random_int'

# lmy
# ACCOUNT_ID = '427502902455'
# IDENTITY_POOL_ID = 'us-east-1:dfc451da-6a3d-447e-93e6-bba055d914fd'
# ROLE_ARN = 'arn:aws:iam::427502902455:role/Cognito_edisonDemoKinesisUnauth_Role'

# lmk
ACCOUNT_ID = '347632544891'
IDENTITY_POOL_ID = 'us-east-1:346c3dd4-2207-4fb2-8644-0bafdaca1a57'
ROLE_ARN = 'arn:aws:iam::347632544891:role/Cognito_edisonDemoKinesisUnauth_Role'

#################################################
# Instantiate cognito and obtain security token #
#################################################
# Use cognito to get an identity.
cognito = boto.connect_cognito_identity()
cognito_id = cognito.get_id(ACCOUNT_ID, IDENTITY_POOL_ID)
oidc = cognito.get_open_id_token(cognito_id['IdentityId'])

# Further setup your STS using the code below
sts = boto.connect_sts()
assumedRoleObject = sts.assume_role_with_web_identity(ROLE_ARN, "XX", oidc['Token'])

# Connect to dynamoDB and kinesis
client_dynamo = boto.dynamodb2.connect_to_region(
	'us-east-1',
	aws_access_key_id=assumedRoleObject.credentials.access_key,
    aws_secret_access_key=assumedRoleObject.credentials.secret_key,
    security_token=assumedRoleObject.credentials.session_token)

client_kinesis = boto.connect_kinesis(
	aws_access_key_id=assumedRoleObject.credentials.access_key,
	aws_secret_access_key=assumedRoleObject.credentials.secret_key,
	security_token=assumedRoleObject.credentials.session_token)

from boto.dynamodb2.table import Table
from boto.dynamodb2.fields import HashKey

######################
# Setup DynamoDB Table
######################

table_dynamo = Table(DYNAMO_TABLE_NAME,connection=client_dynamo)

#################################################
# Setup switch and temperature sensor #
#################################################
# import math
# import pyupm_i2clcd as lcd
# import time

# tempSensor = mraa.Aio(1)
# switch = mraa.Gpio(8)

# B = 4275
# R0 = 100000
# sendToDynamo = True

# myLcd = lcd.Jhd1313m1(0, 0x3E, 0x62)
# myLcd.clear()
# myLcd.setColor(255, 0, 0)
# myLcd.setCursor(0,0)
# myLcd.write('Dynamo')


# def InterruptByButton(args):
# 	global sendToDynamo
	
# 	if sendToDynamo == True:
# 		sendToDynamo = False
# 		print 'Kinesis'

# 		myLcd.clear()
# 		myLcd.setColor(255, 0, 0)
# 		myLcd.setCursor(0,0)
# 		myLcd.write('Kinesis')

# 	else:
# 		sendToDynamo = True
# 		print 'Dynamo'

# 		myLcd.clear()
# 		myLcd.setColor(255, 0, 0)
# 		myLcd.setCursor(0,0)
# 		myLcd.write('Dynamo')


# def GetTemperature():
# 	ADCread = tempSensor.read()
# 	R = 1023.0 / ADCread - 1.0
# 	R = R * R0

# 	temp = 1.0/(math.log10(R/R0)/B+1/298.15)-273.15
# 	return temp

# switch.dir(mraa.DIR_IN)
# switch.isr(mraa.EDGE_RISING, InterruptByButton, InterruptByButton)


try:
	while (1):
		# 	temp = GetTemperature()
		temp = random.randint(1,1000)
		# 	if sendToDynamo == True:
		# 		table_dynamo.put_item({'time':str(time.time()),'temp':str(temp)})
		# 	else:
		package = (str(time.time()),temp)
		client_kinesis.put_record(KINESIS_STREAM_NAME, json.dumps(package), "partitionkey")
		table_dynamo.put_item({'time':str(time.time()),'temp':str(temp)})

		time.sleep(1)

except KeyboardInterrupt:
	exit