import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options
from keywords import GenerateKeyWords
define('port', default=8000, help='run on the given port', type=int)

poi = GenerateKeyWords()


class MainHandler(tornado.web.RequestHandler):

    def get(self):
        title = self.get_argument('title', '这里是一个标题字符串')
        voice = self.get_argument('voice', '这里是一个声音字符串')

        poi.generate(title, voice)
        self.write(str(poi.predicted_poi))


if __name__ == '__main__':
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[(r"/", MainHandler)])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()