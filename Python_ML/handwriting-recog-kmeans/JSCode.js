var mousePressed = false;
var lastX, lastY;
var ctx;

function InitThis() {
  
  // ========= 1
  
    ctx = document.getElementById('myCanvas').getContext("2d");

    $('#myCanvas').mousedown(function (e) {
        mousePressed = true;
        Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
    });

    $('#myCanvas').mousemove(function (e) {
        if (mousePressed) {
            Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
        }
    });

    $('#myCanvas').mouseup(function (e) {
        mousePressed = false;
    });
	    $('#myCanvas').mouseleave(function (e) {
        mousePressed = false;
    });
 
   // =========== 2
  
   ctx2 = document.getElementById('myCanvas2').getContext("2d");

    $('#myCanvas2').mousedown(function (e) {
        mousePressed = true;
        Draw2(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
    });

    $('#myCanvas2').mousemove(function (e) {
        if (mousePressed) {
            Draw2(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
        }
    });

    $('#myCanvas2').mouseup(function (e) {
        mousePressed = false;
    });
	    $('#myCanvas2').mouseleave(function (e) {
        mousePressed = false;
    });
  
  
  // 3==========
  
   ctx3 = document.getElementById('myCanvas3').getContext("2d");

    $('#myCanvas3').mousedown(function (e) {
        mousePressed = true;
        Draw3(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
    });

    $('#myCanvas3').mousemove(function (e) {
        if (mousePressed) {
            Draw3(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
        }
    });

    $('#myCanvas3').mouseup(function (e) {
        mousePressed = false;
    });
	    $('#myCanvas3').mouseleave(function (e) {
        mousePressed = false;
    });
  
  
  // 4 =================
  
   ctx4 = document.getElementById('myCanvas4').getContext("2d");

    $('#myCanvas4').mousedown(function (e) {
        mousePressed = true;
        Draw4(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
    });

    $('#myCanvas4').mousemove(function (e) {
        if (mousePressed) {
            Draw4(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
        }
    });

    $('#myCanvas4').mouseup(function (e) {
        mousePressed = false;
    });
	    $('#myCanvas4').mouseleave(function (e) {
        mousePressed = false;
    });
  
  
  
}




function Draw(x, y, isDown) {
    if (isDown) {
        ctx.beginPath();
        ctx.strokeStyle = $('#selColor').val();
        ctx.lineWidth = $('#selWidth').val();
        ctx.lineJoin = "round";
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.closePath();
        ctx.stroke();
    }
    lastX = x; lastY = y;
}

function Draw2(x, y, isDown) {
    if (isDown) {
        ctx2.beginPath();
        ctx2.strokeStyle = $('#selColor').val();
        ctx2.lineWidth = $('#selWidth').val();
        ctx2.lineJoin = "round";
        ctx2.moveTo(lastX, lastY);
        ctx2.lineTo(x, y);
        ctx2.closePath();
        ctx2.stroke();
    }
    lastX = x; lastY = y;
}

function Draw3(x, y, isDown) {
    if (isDown) {
        ctx3.beginPath();
        ctx3.strokeStyle = $('#selColor').val();
        ctx3.lineWidth = $('#selWidth').val();
        ctx3.lineJoin = "round";
        ctx3.moveTo(lastX, lastY);
        ctx3.lineTo(x, y);
        ctx3.closePath();
        ctx3.stroke();
    }
    lastX = x; lastY = y;
}


function Draw4(x, y, isDown) {
    if (isDown) {
        ctx4.beginPath();
        ctx4.strokeStyle = $('#selColor').val();
        ctx4.lineWidth = $('#selWidth').val();
        ctx4.lineJoin = "round";
        ctx4.moveTo(lastX, lastY);
        ctx4.lineTo(x, y);
        ctx4.closePath();
        ctx4.stroke();
    }
    lastX = x; lastY = y;
}
	
function clearArea() {
    // Use the identity matrix while clearing the canvas
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  
    // clear ctx2
   ctx2.setTransform(1, 0, 0, 1, 0, 0);
    ctx2.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  
      // clear ctx3
   ctx3.setTransform(1, 0, 0, 1, 0, 0);
    ctx3.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  // clear ctx4
  
   ctx4.setTransform(1, 0, 0, 1, 0, 0);
    ctx4.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);


}



function array() {
 
  
   document.getElementById('opening_bracket').innerHTML = "["
  
   var imageData = ctx.getImageData(0, 0, 80, 80);
  
   var data = imageData.data;
  
    for (var i = 0; i < data.length; i += 4) {
      
      var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      data[i]     = avg; // red
      data[i + 1] = avg; // green
      data[i + 2] = avg; // blue
      // ctx.putImageData(imageData, 0, 0);
  };
    
   // logs an array of 10 x 10 x 4 (400 items)
   //document.write(data);
   
  var gray = [];
  
  for (var x = 0; x < data.length; x +=4 ) {
      gray.push(data[x]);
  };
  
  // document.write(gray)
  // 122, 122, 124, 0,   121, 122, 122, 122,   
  //document.write(gray.length)
  // 6400
  
  
  var first_digit = [];
  for (var y = 0; y < data.length; y+=4) {
    first_digit.push(data[y])
  }
 
 
 //document.write(first_digit)
 //document.write(first_digit.length)
 // 6400
  
 var compress = []
 var counter = 0;
 var sum = 0;
 var ten = 0;
 
 for (var z = 0; z < first_digit.length; z++) {
   
   sum = sum + first_digit[z];
   
   if (z % 100 === 0) {
     compress.push(sum/100);
     sum = 0;   
   }
       
 };

  function average(list){
 averageVal = 0
 for(var i = 0; i < list.length; i++){
   averageVal = averageVal + list[i]/list.length
 }
 return averageVal
}

squares = []
for(var i = 0; i < 64; i++){
 squares.push([]);

}

for(var y = 0; y < 80; y++) {

  for(var x = 0; x < 80; x++) {
    
    squares[parseInt(y/10) * 8 + parseInt(x/10)].push(first_digit[x + y * 80])
 
  }
  
}

compressed = []

squares.forEach(function(square){
 compressed.push(average(square)/16)
})

//document.write(compressed)
  
  // round
for (var k = 0; k < compress.length; k++) {
  compressed[k] = compressed[k].toFixed(2);
}
  
  
  document.getElementById('display').innerHTML = "[" + compressed + "]" + ","
  
  
  
  
  
  
  
  // part 2
   var imageData = ctx2.getImageData(0, 0, 80, 80);
  
   var data = imageData.data;
  
    for (var i = 0; i < data.length; i += 4) {
      
      var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      data[i]     = avg; // red
      data[i + 1] = avg; // green
      data[i + 2] = avg; // blue
      // ctx.putImageData(imageData, 0, 0);
  };
    
   // logs an array of 10 x 10 x 4 (400 items)
   //document.write(data);
   
  var gray = [];
  
  for (var x = 0; x < data.length; x +=4 ) {
      gray.push(data[x]);
  };
  
  // document.write(gray)
  // 122, 122, 124, 0,   121, 122, 122, 122,   
  //document.write(gray.length)
  // 6400
  
  
  var first_digit = [];
  for (var y = 0; y < data.length; y+=4) {
    first_digit.push(data[y])
  }
 
 
 //document.write(first_digit)
 //document.write(first_digit.length)
 // 6400
  
 var compress = []
 var counter = 0;
 var sum = 0;
 var ten = 0;
 
 for (var z = 0; z < first_digit.length; z++) {
   
   sum = sum + first_digit[z];
   
   if (z % 100 === 0) {
     compress.push(sum/100);
     sum = 0;   
   }
       
 };

  function average(list){
 averageVal = 0
 for(var i = 0; i < list.length; i++){
   averageVal = averageVal + list[i]/list.length
 }
 return averageVal
}

squares = []
for(var i = 0; i < 64; i++){
 squares.push([]);

}

for(var y = 0; y < 80; y++) {

  for(var x = 0; x < 80; x++) {
    
    squares[parseInt(y/10) * 8 + parseInt(x/10)].push(first_digit[x + y * 80])
 
  }
  
}

compressed = []

squares.forEach(function(square){
 compressed.push(average(square)/16)
})

//document.write(compressed)
  
// round
for (var k = 0; k < compress.length; k++) {
  compressed[k] = compressed[k].toFixed(2);
}
  


  document.getElementById('display2').innerHTML = "[" + compressed + "]" + ","
  
  
  
  
  
  // =============== part 3
  
  var imageData = ctx3.getImageData(0, 0, 80, 80);
  
   var data = imageData.data;
  
    for (var i = 0; i < data.length; i += 4) {
      
      var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      data[i]     = avg; // red
      data[i + 1] = avg; // green
      data[i + 2] = avg; // blue
      // ctx.putImageData(imageData, 0, 0);
  };
    
   // logs an array of 10 x 10 x 4 (400 items)
   //document.write(data);
   
  var gray = [];
  
  for (var x = 0; x < data.length; x +=4 ) {
      gray.push(data[x]);
  };
  
  // document.write(gray)
  // 122, 122, 124, 0,   121, 122, 122, 122,   
  //document.write(gray.length)
  // 6400
  
  
  var first_digit = [];
  for (var y = 0; y < data.length; y+=4) {
    first_digit.push(data[y])
  }
 
 
 //document.write(first_digit)
 //document.write(first_digit.length)
 // 6400
  
 var compress = []
 var counter = 0;
 var sum = 0;
 var ten = 0;
 
 for (var z = 0; z < first_digit.length; z++) {
   
   sum = sum + first_digit[z];
   
   if (z % 100 === 0) {
     compress.push(sum/100);
     sum = 0;   
   }
       
 };

  function average(list){
 averageVal = 0
 for(var i = 0; i < list.length; i++){
   averageVal = averageVal + list[i]/list.length
 }
 return averageVal
}

squares = []
for(var i = 0; i < 64; i++){
 squares.push([]);

}

for(var y = 0; y < 80; y++) {

  for(var x = 0; x < 80; x++) {
    
    squares[parseInt(y/10) * 8 + parseInt(x/10)].push(first_digit[x + y * 80])
 
  }
  
}

compressed = []

squares.forEach(function(square){
 compressed.push(average(square)/16)
})

//document.write(compressed)
  
// round
for (var k = 0; k < compress.length; k++) {
  compressed[k] = compressed[k].toFixed(2);
}
  
  document.getElementById('display3').innerHTML = "[" + compressed + "]" + ","
  
  
  
  
  
  
  
  // =========== 4
  var imageData = ctx4.getImageData(0, 0, 80, 80);
  
   var data = imageData.data;
  
    for (var i = 0; i < data.length; i += 4) {
      
      var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      data[i]     = avg; // red
      data[i + 1] = avg; // green
      data[i + 2] = avg; // blue
      // ctx.putImageData(imageData, 0, 0);
  };
    
   // logs an array of 10 x 10 x 4 (400 items)
   //document.write(data);
   
  var gray = [];
  
  for (var x = 0; x < data.length; x +=4 ) {
      gray.push(data[x]);
  };
  
  // document.write(gray)
  // 122, 122, 124, 0,   121, 122, 122, 122,   
  //document.write(gray.length)
  // 6400
  
  
  var first_digit = [];
  for (var y = 0; y < data.length; y+=4) {
    first_digit.push(data[y])
  }
 
 
 //document.write(first_digit)
 //document.write(first_digit.length)
 // 6400
  
 var compress = []
 var counter = 0;
 var sum = 0;
 var ten = 0;
 
 for (var z = 0; z < first_digit.length; z++) {
   
   sum = sum + first_digit[z];
   
   if (z % 100 === 0) {
     compress.push(sum/100);
     sum = 0;   
   }
       
 };

  function average(list){
 averageVal = 0
 for(var i = 0; i < list.length; i++){
   averageVal = averageVal + list[i]/list.length
 }
 return averageVal
}

squares = []
for(var i = 0; i < 64; i++){
 squares.push([]);

}

for(var y = 0; y < 80; y++) {

  for(var x = 0; x < 80; x++) {
    
    squares[parseInt(y/10) * 8 + parseInt(x/10)].push(first_digit[x + y * 80])
 
  }
  
}

compressed = []

squares.forEach(function(square){
 compressed.push(average(square)/16)
})

// round
for (var k = 0; k < compress.length; k++) {
  compressed[k] = compressed[k].toFixed(2);
}
  
//document.write(compressed)
  
  
  document.getElementById('display4').innerHTML = "[" + compressed + "]"
  
  document.getElementById('closing_bracket').innerHTML = "]"

}

