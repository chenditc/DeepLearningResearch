export http_proxy=http://chenditc:Proxymesh013001@us-il.proxymesh.com:31280
mkdir /home/ubuntu/news_data/crawler_job/hexun
nohup scrapy crawl hexun -s JOBDIR=/home/ubuntu/news_data/crawler_job/hexun > /home/ubuntu/news_data/crawl_log &
