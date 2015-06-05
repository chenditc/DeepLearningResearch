mkdir /home/ubuntu/data
mkdir /home/ubuntu/crawljob
mkdir /home/ubuntu/crawljob/i7wu
nohup scrapy crawl i7wu -s JOBDIR=/home/ubuntu/crawljob/i7wu/ &
