/*
 * Copyright 2013-2014 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Amazon Software License (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 * http://aws.amazon.com/asl/
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

package wordStream;


import backtype.storm.topology.BasicOutputCollector;
import backtype.storm.topology.base.BaseBasicBolt;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.tuple.Tuple;

public class PrintBolt extends BaseBasicBolt {

    private static final Logger LOG = LoggerFactory.getLogger(PrintBolt.class);
    

    @Override
    public void execute(Tuple input,  BasicOutputCollector collector) {
        LOG.info(input.toString());
    }


    @Override
    public void declareOutputFields(OutputFieldsDeclarer arg0) {
        // TODO Auto-generated method stub
        
    }

}