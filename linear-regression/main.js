let xpoints = [];
let ypoints = [];

let m, c;

const learningRate = 0.2;
const optimizer = tf.train.sgd(learningRate);

const loss = (pred, label) => pred.sub(label).square().mean();

function setup() {
    createCanvas(window.innerWidth, window.innerHeight);

    m = tf.variable(tf.scalar(random(1)));
    c = tf.variable(tf.scalar(random(1)));
}

function draw() {

    background(10);

    stroke(255);
    strokeWeight(6);
    for (let i in xpoints) {
        point(xpoints[i]*width, (1 - ypoints[i])*height);
    }

    if (xpoints.length) {
		optimizer.minimize(() => loss(predict(xpoints), tf.tensor1d(ypoints)) );
	
		let x = [0,1];
		let y = tf.tidy(() =>predict(x).dataSync());
        strokeWeight(4);
		line(x[0]*width,(1 - y[0])*height,x[1]*width,(1-y[1])*height);
	}

}


function predict(x) {
    x_t = tf.tensor1d(x);
    y_t = x_t.mul(m).add(c);

    return y_t;
}

function mousePressed() {
    xpoints.push(mouseX/width);
    ypoints.push(1 - mouseY/height);
}
