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


import java.util.Map;

import backtype.storm.task.ShellBolt;
import backtype.storm.topology.IRichBolt;

import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.tuple.Fields;

public class UpdateGradientBolt extends ShellBolt implements IRichBolt {

    public UpdateGradientBolt() {
      super("/bin/storm/UpdateGradient");
    }   

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
      
    }   
    

    @Override
    public Map<String, Object> getComponentConfiguration() {
      return null;
    }   
  }