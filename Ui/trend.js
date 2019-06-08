<head>
 <meta charset='utf-8'>
</head>
<svg viewBox="0 0 500 200" class="chart">
<text x="85" y="28" dy=".35em">I Am A Trend Title &#x1F4A9;</text>
<title id="title">Its totally cool \u1F4A9;</title>
  <polyline class="line_plot"
     fill="none"
     stroke="#0074d9"
     stroke-width="3"
     points="
       0,0
       1,1"/>
       
</svg>


setInterval(function(e) {
	lineNode = document.getElementsByClassName("line_plot")[0];
  points = lineNode.getAttribute("points").trim();
  pairs = points.replace(' ', '').split('\n');
  //console.log("pairs: "+pairs);
  xyPairs = [];
  for(var i = 0; i < pairs.length; i++){
  	pair = pairs[i].split(',');
    x = Number(pair[0]);
    y = Number(pair[1]);
		xyPairs.push([x,y])
  }
  last = xyPairs[xyPairs.length-1];
  xyPairs.push([last[0]+1,last[1]+1]);
  //console.log("xyPairs: "+xyPairs);
  newPoints = xyPairs.map(function(pair){return pair[0].toString()+","+pair[1].toString()}).join('\n');
  //console.log("new points: "+newPoints);
  lineNode.setAttribute("points",newPoints);
  
},1000);
