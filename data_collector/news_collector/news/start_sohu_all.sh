export http_proxy=http://chenditc:Proxymesh013001@us-il.proxymesh.com:31280
mkdir /home/ubuntu/news_data/crawler_job/sohu_all
nohup scrapy crawl sohu_all -s JOBDIR=/home/ubuntu/news_data/crawler_job/sohu_all > /home/ubuntu/news_data/crawl_log_sohu_all &
