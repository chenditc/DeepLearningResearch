export http_proxy=http://chenditc:Proxymesh013001@us-il.proxymesh.com:31280
mkdir /home/ubuntu/news_data/crawler_job/sohu
nohup scrapy crawl sohu -s JOBDIR=/home/ubuntu/news_data/crawler_job/sohu > /home/ubuntu/news_data/crawl_log_sohu &
