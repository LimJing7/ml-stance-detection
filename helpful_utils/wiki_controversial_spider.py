from twisted.internet import reactor
import scrapy
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging

class MySpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            'https://en.wikipedia.org/wiki/Wikipedia:List_of_controversial_issues'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        all_output = []
        page = response.url.split("/")[-2]
        filename = f'controversial.txt'
        values = response.xpath('//li')
        for val in values:
            output = ''
            child_nodes = val.xpath('./child::node()')
            for node in child_nodes:
                text = node.xpath('./text()').get()
                if text is not None:
                    output += text
                else:
                    output += node.get()
            all_output.append(output)
        with open(filename, 'w') as f:
            for line in all_output:
                f.write(line)
                f.write('\n')
        self.log(f'Saved file {filename}')

configure_logging({'LOG_FORMAT': '%(levelname)s: %(message)s'})
runner = CrawlerRunner()

d = runner.crawl(MySpider)
d.addBoth(lambda _: reactor.stop())
reactor.run() # the script will block here until the crawling is finished
