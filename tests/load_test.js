import http from 'k6/http';


const image = open('./test.jpg', 'b')
export default function () {
    const url = 'http://localhost:8002/predict';
    const data = {
        image: http.file(image, 'test.jpg'),
    };

    http.post(url, data);
}