let xpoints = [];
let ypoints = [];

let a = [];
let n = 10;

const learningRate = 0.1;
const optimizer = tf.train.adam(learningRate);

const loss = (pred, label) => pred.sub(label).square().mean();

function setup() {
    createCanvas(window.innerWidth, window.innerHeight);

    for(let i=0;i<n;i++){
        a.push(tf.variable(tf.scalar(random(-1,1))));
    }
}

function draw() {
    background(10);

    stroke(255);
    strokeWeight(6);
    for (let i in xpoints) {
        let x_temp = map(xpoints[i],-1,1,0,width);
        let y_temp = map(ypoints[i],-1,1,height,0);
        point(x_temp, y_temp);
    }

    if (xpoints.length) {
		optimizer.minimize(() => loss(predict(xpoints), tf.tensor1d(ypoints)) );
    
        let x = [];
        for(let i=-1;i<=1;i+=0.1)
            x.push(i);
            
		let y = tf.tidy(() =>predict(x).dataSync());
        strokeWeight(4);

        for (let i=0;i<x.length -1;i++) {
            let x_temp = map(x[i],-1,1,0,width);
            let y_temp = map(y[i],-1,1,height,0);
            let x2_temp = map(x[i+1],-1,1,0,width);
            let y2_temp = map(y[i+1],-1,1,height,0);
            line(x_temp, y_temp,x2_temp, y2_temp);
        }
	}

}


function predict(x) {
    const x_t = tf.tensor1d(x);
    let result,y_t = a[n-1];;

    for (let index = 1; index < n; index++) {
        if(index == n-1){
            y_t = y_t.add(x_t.pow(tf.tensor1d([index])).mul(a[n-index-1]));               
            result = y_t;
        }else
            y_t = y_t.add(x_t.pow(tf.tensor1d([index])).mul(a[n-index-1]));   
        
    }
    return result;
}

function mousePressed() {
    let x_temp = map(mouseX,0,width,-1,1);
    let y_temp = map(mouseY,0,height,1,-1);
    xpoints.push(x_temp);
    ypoints.push(y_temp);
}
