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

import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.Charset;
import java.nio.charset.CharsetDecoder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import backtype.storm.task.TopologyContext;
import backtype.storm.topology.BasicOutputCollector;
import backtype.storm.topology.base.BaseBasicBolt;

import com.amazonaws.services.kinesis.stormspout.DefaultKinesisRecordScheme;
import com.amazonaws.util.json.JSONArray;
import com.amazonaws.util.json.JSONException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import redis.clients.jedis.Jedis;

import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.tuple.Tuple;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Values;
import com.amazonaws.util.json.JSONObject;
import com.amazonaws.services.kinesis.model.Record;

public class ProjectionLayerBolt extends BaseBasicBolt {

    private static final Logger LOG = LoggerFactory.getLogger(ProjectionLayerBolt.class);
    private static final CharsetDecoder decoder = Charset.forName("UTF-8").newDecoder();
    private Jedis redisClient;
    private int wordEmbeddingDimension = 50;
    private String redisEndpoint;
    
    
    public ProjectionLayerBolt(String redisEndpoint) {
        this.redisEndpoint = redisEndpoint;
    }
    
    @Override
    public void prepare(Map stormConf, TopologyContext context)
    {
        // Initialize redis client
        redisClient = new Jedis(this.redisEndpoint);
    }
    
    public String getEmbeddingFromRedis(String key) {
        String embeddingString = redisClient.get(key);
        if (embeddingString == null) {
            embeddingString = "[0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123, 0.9182987219387123]";
            redisClient.set(key, embeddingString);
        }
        return embeddingString;
    }

    @Override
    public void execute(Tuple input,  BasicOutputCollector collector) {
        Record record = (Record)input.getValueByField(DefaultKinesisRecordScheme.FIELD_RECORD);
        ByteBuffer buffer = record.getData();
        String data = null; 
        try {
            data = decoder.decode(buffer).toString();
            
            LOG.info(data);
            
            JSONObject jsonObject = new JSONObject(data);

            JSONArray indexOfXArray = jsonObject.getJSONArray("x");
            String indexOfX = jsonObject.getString("x");
            String indexOfY = jsonObject.getString("y");
            HashMap<String, JSONArray> wordEmbedding = new HashMap<>();
            
            for (int i = 0; i < indexOfXArray.length(); i++) {
                int index_x = (int) indexOfXArray.get(i);
                String key = Integer.toString(index_x);
                String embeddingString = getEmbeddingFromRedis(key);
                wordEmbedding.put(key, new JSONArray(embeddingString));
            }
            JSONObject wordEmbeddingObject = new JSONObject(wordEmbedding);
            
            LOG.info(wordEmbeddingObject.toString());

            collector.emit(new Values(indexOfX, indexOfY, wordEmbeddingObject));

        } catch (CharacterCodingException|JSONException|IllegalStateException e) {
            LOG.error("Exception when decoding record ", e);
        }
    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("vector_x", "vector_y", "wordEmbedding"));
    }

}