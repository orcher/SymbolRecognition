import QtQuick 2.4
import QtQuick.Window 2.2

Window {
	width: 600
	height: 400
    visible: true

    Text{
        text: net.error
        font.family: "Helvetica"
        font.pointSize: 24
        color: "red"

        anchors.centerIn: parent
    }
}
