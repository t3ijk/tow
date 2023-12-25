fs = require('fs')
lines = fs.readFileSync('./nohup.out').toString('utf8').split('\n')
console.log(lines.length)
nl = lines.length

const average = arr => arr.reduce((p, c) => Number(p) + Number(c), 0) / arr.length;

String.prototype.replaceAll = function (find, replace) {
    var str = this;
    return str.replace(new RegExp(find, 'g'), replace);
};

flag1 = true
lossList = []
newLines = []

for (var i = 0; i < nl; i++) {
    if (lossList.length === 30) {
        mean = average(lossList)
    }
    obj = null
    try {
        obj1 = JSON.parse(lines[i].replaceAll("'", '"'))
        if (Array.isArray(obj1) && obj1.length == 2) {
            if (lossList.length === 30) {
                lossList = []
            }
            lossList.push(obj1[0])
        }

        obj2 = JSON.parse(lines[i])
        if (obj2['ws'] == 6) {
            obj2['vls'] = mean
            // console.log('loss mean ', mean)
            newLines.push(JSON.stringify(obj2))
        } else {
            newLines.push(lines[i])
        }

    } catch(e) {
        newLines.push(lines[i])
    }

}

fs.writeFileSync('./nohup.out.fix', newLines.join('\n'))
